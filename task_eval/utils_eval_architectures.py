import random
import time

from dotenv import load_dotenv
from groq import Groq
import os
import json
import re
import requests
import tiktoken
from tqdm import tqdm
from global_methods import run_llama_open_router, run_llama_groq
from memory_architectures import (
    EXPERIMENTS,
    MemoryBundle,
    VectorStore,
    build_memory,
    flatten_turns,
    persist_bundle,
    retrieve_context,
)

load_dotenv()

# cl100k_base is a generic approximation, not each model's real tokenizer
# (Qwen/Llama/etc. use their own vocabularies) -- but it's a consistent,
# documented, reproducible choice across every model in the study, which is
# what a controlled Axis B (sliding-window) condition needs. Loaded lazily
# and cached module-wide so repeated calls don't re-init the encoder.
_TIKTOKEN_ENCODER = None


def _get_tiktoken_encoder():
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODER


def count_tokens(text):
    return len(_get_tiktoken_encoder().encode(text))


MAX_LENGTH = {"claude-sonnet": 2000000, "claude-haiku": 2000000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):
    # single_quote_count = text.count("'")
    # double_quote_count = text.count('"')
    # if single_quote_count > double_quote_count:
    #     text = text.replace('"', "")
    #     text = text.replace("'", '"')
    #     print(text)
    text = text.strip()
    if text[0] != "{":
        start = text.index("{")
        text = text[start:].strip()

    return json.loads(text)


def truncation_sliding_window(
    raw_convo_json: str, context_window_limit: int, tokenizer
):
    # JSON validation
    try:
        convo_json = json.loads(raw_convo_json)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON input") from e

    # Basic argument validation
    if not isinstance(context_window_limit, int) or context_window_limit <= 0:
        raise ValueError("context_window_limit must be a positive integer")

    # Extract conversation object
    try:
        conversation = convo_json["conversation"]
    except (TypeError, KeyError) as e:
        raise ValueError("Missing required 'conversation' key in JSON input") from e

    # Extract sessions in chronological order
    sessions = []

    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time"):
            if isinstance(value, list):
                sessions.append(value)

    if not sessions:
        raise ValueError("No conversation sessions found")

    # Flatten all dialogue turns while preserving chronological order
    turns = []

    for session in sessions:
        for turn in session:
            if not isinstance(turn, dict):
                continue

            speaker = turn.get("speaker")
            text = turn.get("text")

            if speaker is None or text is None:
                continue

            turns.append(
                {"speaker": speaker, "text": text, "dia_id": turn.get("dia_id")}
            )

    if not turns:
        raise ValueError("No valid dialogue turns found")

    # Start from the newest turn and work backwards
    selected_turns = []
    token_count = 0

    for turn in reversed(turns):
        formatted_turn = f'{turn["speaker"]}: {turn["text"]}'

        # Tokenize this turn
        turn_tokens = tokenizer.encode(formatted_turn, add_special_tokens=False)

        turn_token_count = len(turn_tokens)

        # Stop once adding an older turn would exceed the budget
        if token_count + turn_token_count > context_window_limit:
            break

        selected_turns.append(turn)
        token_count += turn_token_count

    # Restore chronological order
    selected_turns.reverse()

    return {
        "turns": selected_turns,
        "formatted_context": "\n".join(
            f'{turn["speaker"]}: {turn["text"]}' for turn in selected_turns
        ),
        "token_count": token_count,
    }


def set_groq_api_key():
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_input_context(data, num_question_tokens, model, args, max_context_tokens=None):
    session_nums = [
        int(k.split("_")[-1])
        for k in data.keys()
        if "session" in k and "date_time" not in k
    ]

    # Build each session as its own block, oldest first, exactly as before.
    session_blocks = []
    for i in range(min(session_nums), max(session_nums) + 1):
        if "session_%s" % i not in data:
            continue
        session_text = ""
        for dialog in data["session_%s" % i][::-1]:
            turn = dialog["speaker"] + ' said, "' + dialog["text"] + '"' + "\n"
            if "blip_caption" in dialog:
                turn += " and shared %s." % dialog["blip_caption"]
            turn += "\n"
            session_text = turn + session_text

        session_text = (
            "\nDATE: "
            + data["session_%s_date_time" % i]
            + "\n"
            + "CONVERSATION:\n"
            + session_text
        )
        session_blocks.append(session_text)

    if max_context_tokens is not None:
        # Axis B = "truncated / sliding window": keep the most recent
        # sessions that fit the budget (real tiktoken counts, not a
        # heuristic), dropping the oldest ones first. Reserve room for the
        # question(s)/prompt overhead via num_question_tokens. Always keep
        # at least the single most recent session so a query never goes out
        # with zero context, even if that one session alone exceeds budget.
        budget = max(max_context_tokens - num_question_tokens, 0)
        kept = []
        running_tokens = 0
        for block in reversed(session_blocks):
            block_tokens = count_tokens(block)
            if kept and running_tokens + block_tokens > budget:
                break
            kept.append(block)
            running_tokens += block_tokens
        kept.reverse()
        session_blocks = kept
    # else: max_context_tokens is None -> Axis B = "full long-context",
    # every session is kept, no truncation at all.

    return "\n\n".join(session_blocks)


def get_cat_5_answer(model_prediction, answer_key):
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if "a" in model_prediction:
            return answer_key["a"]
        else:
            return answer_key["b"]
    elif len(model_prediction) == 3:
        if "(a)" in model_prediction:
            return answer_key["a"]
        else:
            return answer_key["b"]
    else:
        return model_prediction



def _artifact_run_key(retriever):
    """Return a filesystem-safe cache namespace for one memory-writing setup.

    Memory artifacts are generated by the evaluated model.  They must never be
    reused by a different model/provider/retriever combination, or the result
    would measure one model answering from another model's memories.
    """
    def clean(value):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._") or "default"

    return "__".join((clean(retriever), "v1"))


def _compact_turn(turn):
    """Persist retrieval provenance without duplicating conversation text."""
    return {
        "session_idx": turn.get("session_idx"),
        "date": turn.get("date"),
        "dia_id": turn.get("dia_id"),
        "speaker": turn.get("speaker"),
    }


def _load_memory_bundle(
    artifact_dir,
    sample_id,
    experiment,
    conversation,
    model,
    retriever,
    llm_fn=None,
):
    """Load or build artifacts for exactly one reproducible memory-writing run."""
    from pathlib import Path
    base = Path(artifact_dir) / run_key / str(sample_id) / experiment

    def cached_metadata():
        metadata_path = base / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Cached memory bundle is missing metadata: {metadata_path}")
        metadata = json.load(open(metadata_path, encoding="utf-8"))
        return metadata

    if experiment == "summary_rag" and (base / "vector_store.json").exists():
        store = VectorStore.load(str(base / "vector_store.json"))
        items = json.load(open(base / "items.json", encoding="utf-8"))
        return MemoryBundle(experiment, "summaries", "vector", {"items": items, "store": store}, cached_metadata())
    if experiment == "facts_rag" and (base / "vector_store.json").exists():
        store = VectorStore.load(str(base / "vector_store.json"))
        items = json.load(open(base / "items.json", encoding="utf-8"))
        return MemoryBundle(experiment, "facts", "vector", {"items": items, "store": store}, cached_metadata())
    if experiment == "summary_tiered" and (base / "summary_vector_store.json").exists():
        store = VectorStore.load(str(base / "summary_vector_store.json"))
        items = json.load(open(base / "summary_items.json", encoding="utf-8"))
        return MemoryBundle(
            experiment, "summaries+raw", "tiered",
            {"summary_items": items, "summary_store": store,
             "raw_turns": flatten_turns(conversation)},
            cached_metadata(),
        )
    if experiment == "graph_traversal" and (base / "graph.json").exists():
        graph = json.load(open(base / "graph.json", encoding="utf-8"))
        return MemoryBundle(experiment, "graph", "graph", {"graph": graph}, cached_metadata())

    bundle = build_memory(
        conversation,
        experiment=experiment,
        retriever=retriever,
        artifact_dir=artifact_dir,
        llm_fn=llm_fn,
    )
    persist_bundle(bundle, artifact_dir, str(sample_id))
    return bundle


def get_llama_answers(in_data, out_data, prediction_key, args):
    """Run one of the six LoCoMo-adapted memory conditions over a sample."""
    llm_fn = getattr(args, "llm_fn", None)
    if llm_fn is None:
        raise ValueError("Local evaluation requires args.llm_fn with the standard model-call signature.")
    assert len(in_data["qa"]) == len(out_data["qa"]), (
        len(in_data["qa"]),
        len(out_data["qa"]),
    )

    experiment = getattr(args, "experiment", None) or "raw_truncated"
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    speakers_names = sorted(set(d["speaker"] for d in in_data["conversation"]["session_1"]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])

    # Raw conditions do not require a memory-generation LLM call.
    bundle = None
    if experiment not in {"raw_truncated", "raw_full"}:
        bundle = _load_memory_bundle(
            getattr(args, "emb_dir", "") or getattr(args, "memory_dir", "memory_cache"),
            in_data["sample_id"],
            experiment,
            in_data["conversation"],
            args.model,
            getattr(args, "retriever", "contriever"),
            llm_fn=llm_fn,
        )

    # Use the reproducible tokenizer already adopted by the sliding-window condition.
    tokenizer = None
    if experiment in {"raw_truncated", "raw_full"} or experiment == "summary_tiered":
        try:
            tokenizer = _get_tiktoken_encoder()
        except Exception:
            tokenizer = None

    qa_items = in_data["qa"]
    max_questions = getattr(args, "max_questions", None)
    if max_questions is not None:
        qa_items = qa_items[:max_questions]

    for i, qa in enumerate(qa_items):
        if prediction_key in out_data["qa"][i] and not args.overwrite:
            continue

        question = qa["question"]
        if qa["category"] == 2:
            question = question + " Use DATE of CONVERSATION to answer with an approximate date."

        cat5_answer = None
        if qa["category"] == 5:
            if random.random() < 0.5:
                question = question + " Select the correct answer: (a) Not mentioned in the conversation (b) " + qa["answer"] + "."
                cat5_answer = {"a": "Not mentioned in the conversation", "b": qa["answer"]}
            else:
                question = question + " Select the correct answer: (a) " + qa["answer"] + " (b) Not mentioned in the conversation."
                cat5_answer = {"a": qa["answer"], "b": "Not mentioned in the conversation"}

        if experiment in {"raw_truncated", "raw_full"}:
            context, trace = retrieve_context(
                MemoryBundle(experiment, "raw", "truncated" if experiment == "raw_truncated" else "full", {}, dict(EXPERIMENTS[experiment])),
                in_data["conversation"],
                question,
                tokenizer=tokenizer,
                context_budget=args.context_token_budget,
                top_k=args.top_k,
                tier_working_turns=args.tier_working_turns,
            )
        else:
            context, trace = retrieve_context(
                bundle,
                in_data["conversation"],
                question,
                tokenizer=tokenizer,
                context_budget=args.context_token_budget,
                top_k=args.top_k,
                tier_working_turns=args.tier_working_turns,
            )

        query = start_prompt + "\n" + context + "\n\n" + (
            QA_PROMPT_CAT_5.format(question) if qa["category"] == 5 else QA_PROMPT.format(question)
        )

        answer = llm_fn(query, PER_QA_TOKEN_BUDGET, api_key=os.getenv("GROQ_API_KEY"), model=args.model)
        answer = answer.strip()
        if cat5_answer is not None:
            answer = get_cat_5_answer(answer, cat5_answer)

        out_data["qa"][i][prediction_key] = answer
        out_data["qa"][i][prediction_key + "_trace"] = {
            "experiment": experiment,
            "access": EXPERIMENTS[experiment]["access"],
            "storage": EXPERIMENTS[experiment]["storage"],
            "context_tokens": trace.get("token_count"),
            "truncated_to_budget": trace.get("truncated_to_budget"),
        }

        # Keep retrieval provenance separate from the answer so the benchmark score
        # is unaffected, while making the experiment auditable.
        if args.log_memory_trace:
            if "accessed" in trace:
                accessed = trace["accessed"]
                out_data["qa"][i][prediction_key + "_trace"]["accessed"] = (
                    "all_raw_turns"
                    if accessed == "all_raw_turns"
                    else [_compact_turn(turn) for turn in accessed]
                )
            if "working_set" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["working_set"] = [
                    _compact_turn(turn) for turn in trace["working_set"]
                ]
            if "paged_turns" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["paged_turns"] = [
                    _compact_turn(turn) for turn in trace["paged_turns"]
                ]
            if "retrieved" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["retrieved"] = [
                    {k: v for k, v in item.items() if k != "text"} for item in trace["retrieved"]
                ]
            if "recalled" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["recalled"] = [
                    {k: v for k, v in item.items() if k != "text"} for item in trace["recalled"]
                ]
            if "paged_sessions" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["paged_sessions"] = trace["paged_sessions"]
            if "graph_facts" in trace:
                out_data["qa"][i][prediction_key + "_trace"]["graph_facts"] = trace["graph_facts"]

    out_data.setdefault("run_metadata", {})[prediction_key] = {
        "experiment": experiment,
        "retriever": getattr(args, "retriever", None),
        "context_token_budget": args.context_token_budget,
        "top_k": args.top_k,
        "tier_working_turns": args.tier_working_turns,
        "seed": getattr(args, "seed", None),
        "artifact_cache_key": _artifact_run_key(
            getattr(args, "retriever", "contriever")
        ),
    }

    return out_data
