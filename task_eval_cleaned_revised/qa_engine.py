"""QA generation engine: sliding-window context, batching, category-5 handling, and RAG retrieval."""

import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

import tiktoken
import numpy as np
import requests
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVIDIA Embedding API (self-contained) - UPDATED for nemotron-3-embed-1b
# ---------------------------------------------------------------------------
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
EMBED_MODEL = os.environ.get(
    "NVIDIA_EMBED_MODEL", "nvidia/nemotron-3-embed-1b"
)  # UPDATED model name


def get_embeddings(texts: List[str], mode: str = "context") -> np.ndarray:
    """get embeddings model via nvidia api"""
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    input_type = "passage" if mode == "context" else "query"

    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
    }
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings", json=payload, headers=headers
    )
    response.raise_for_status()
    data = response.json()
    return np.array([item["embedding"] for item in data["data"]])


# LAZY SINGLETOOOON!
_TIKTOKEN_ENCODER = None


def get_tokenizer():
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODER


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text))


CONV_START_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

QA_PROMPT_SINGLE = (
    "Based on the above context, write an answer in the form of a short phrase "
    "for the following question. Answer with exact words from the context "
    "whenever possible.\n\nQuestion: {question} Short answer:"
)

QA_PROMPT_CAT5 = (
    "Based on the above context, answer the following question.\n\n"
    "Question: {question} Short answer:"
)

QA_PROMPT_BATCH = (
    "Based on the above conversations, write short answers for each of the "
    "following questions in a few words. Write the answers in the form of a "
    "JSON dictionary where each entry contains the string format of question "
    "number as 'key' and the short answer as value. Use single-quote characters "
    "for named entities. Answer with exact words from the conversations whenever "
    "possible.\n\n"
)

PER_QA_TOKEN_BUDGET = 50
START_PROMPT_OVERHEAD_TOKENS = 100


def retrieve_context(in_data: Dict, args: Any):

    conversation = in_data["conversation"]

    session_nums = [
        int(k.split("_")[-1])
        for k in conversation.keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    ]

    session_nums = sorted(session_nums)

    texts = []
    ids = []
    dates = []

    for i in session_nums:
        if args.rag_mode == "summary":
            field = f"session_{i}_summary"

            if field in conversation:
                texts.append(conversation[field])

                ids.append(f"S{i}")
                dates.append(conversation.get(f"session_{i}_date_time", "Unknown"))

        elif args.rag_mode == "observation":
            field = f"session_{i}_observation"
            if field in conversation:
                texts.append(conversation[field])
                ids.append(f"S{i}")
                dates.append(conversation.get(f"session_{i}_date_time", "Unknown"))

        elif args.rag_mode == "dialog":
            if f"session_{i}" in conversation:
                block = (
                    f"\nDATE: {conversation.get(f'session_{i}_date_time', 'Unknown')}\n"
                )
                for turn in conversation[f"session_{i}"]:
                    line = f'{turn["speaker"]} said, "{turn["text"]}"\n'
                    if "blip_caption" in turn:
                        line += f" and shared {turn['blip_caption']}.\n"
                    block += line
                texts.append(block)
                ids.append(f"S{i}")
                dates.append(conversation.get(f"session_{i}_date_time", "Unknown"))

    if not texts:
        return None, None, None, None

    context_embeddings = get_embeddings(texts, mode="context")

    return context_embeddings, texts, ids, dates


def retrieve_top_k(
    question: str,
    context_embeddings: np.ndarray,
    texts: List[str],
    ids: List[str],
    dates: List[str],
    top_k: int,
):
    """
    embed the question,compute similarity, and return the topk context chunks.
    """
    # Embed the query - mode='query' -> input_type='query'
    query_emb = get_embeddings([question], mode="query")  # shape (1, dim)
    sims = np.dot(context_embeddings, query_emb.T).squeeze()

    top_indices = np.argsort(sims)[-top_k:][::-1]

    retrieved_texts = [texts[idx] for idx in top_indices]
    retrieved_ids = [ids[idx] for idx in top_indices]
    retrieved_dates = [dates[idx] for idx in top_indices]

    return retrieved_texts, retrieved_ids, retrieved_dates


def build_context(
    conversation: Dict[str, Any],
    reserved_tokens: int,
    max_context_tokens: Optional[int],
) -> str:

    session_nums = [
        int(k.split("_")[-1])
        for k in conversation.keys()
        if k.startswith("session_") and not k.endswith("_date_time")
    ]

    blocks: List[str] = []
    for i in range(min(session_nums), max(session_nums) + 1):
        key = f"session_{i}"
        if key not in conversation:
            continue

        turns_text = ""
        for turn in reversed(conversation[key]):
            line = f'{turn["speaker"]} said, "{turn["text"]}"\n'
            if "blip_caption" in turn:
                line += f" and shared {turn['blip_caption']}."
            line += "\n"
            turns_text = line + turns_text

        date = conversation.get(f"session_{i}_date_time", "Unknown")
        blocks.append(f"\nDATE: {date}\nCONVERSATION:\n{turns_text}")

    if max_context_tokens is not None:
        budget = max(max_context_tokens - reserved_tokens, 0)
        kept: List[str] = []
        running = 0
        for block in reversed(blocks):
            tokens = count_tokens(block)
            if kept and running + tokens > budget:
                break
            kept.append(block)
            running += tokens
        kept.reverse()
        blocks = kept

    return "\n\n".join(blocks)


# helper funcs for cat-5
def build_cat5_question(question: str, answer: str) -> tuple[str, Dict[str, str]]:
    """Shuffle (a)/(b) order for category-5 and return the answer key."""
    if random.random() < 0.5:
        text = f"{question} Select the correct answer: (a) Not mentioned in the conversation (b) {answer}."
        key = {"a": "Not mentioned in the conversation", "b": answer}
    else:
        text = f"{question} Select the correct answer: (a) {answer} (b) Not mentioned in the conversation."
        key = {"a": answer, "b": "Not mentioned in the conversation"}
    return text, key


def resolve_cat5(prediction: str, answer_key: Dict[str, str]) -> str:
    """map each of the short multiple choice prediction to the full answer text in ds"""
    pred = prediction.strip().lower()
    if len(pred) == 1:
        return answer_key.get("a" if "a" in pred else "b", prediction)
    if "(a)" in pred:
        return answer_key.get("a", prediction)
    if "(b)" in pred:
        return answer_key.get("b", prediction)
    return prediction


PARSE_ERRORS = (json.JSONDecodeError, ValueError, IndexError, KeyError)


def parse_json_output(raw: str) -> Dict[str, Any]:
    """Extract a JSON dict from model output, stripping markdown fences if present."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.replace("json", "").replace("`", "").strip()

    if not text.startswith("{"):
        start = text.index("{")
        text = text[start:]

    return json.loads(text)


def generate_answers(
    in_data: Dict, out_data: Dict, prediction_key: str, args: Any
) -> Dict:
    """make predictions for every QA item in a single sample."""
    llm_fn = getattr(args, "llm_fn", None)
    if llm_fn is None:
        raise ValueError(
            "args.llm_fn must be bound to a callable(query, max_tokens) -> str"
        )

    assert len(in_data["qa"]) == len(out_data["qa"]), (
        "QA length mismatch between input and output"
    )

    session_1 = in_data["conversation"].get("session_1", [])
    speakers = sorted({t["speaker"] for t in session_1})
    if len(speakers) < 2:
        speakers += ["Unknown"] * (2 - len(speakers))
    start_prompt = CONV_START_PROMPT.format(
        speaker_a=speakers[0], speaker_b=speakers[1]
    )

    qa_items = in_data["qa"]
    if getattr(args, "max_questions", None) is not None:
        qa_items = qa_items[: args.max_questions]

    # Pre‑compute retrieval database if using RAG
    if args.use_rag:
        context_embeddings, texts, ids, dates = retrieve_context(in_data, args)
    else:
        context_embeddings = texts = ids = dates = None

    for batch_start in tqdm(
        range(0, len(qa_items), args.batch_size), desc="Generating answers"
    ):
        batch_questions: List[str] = []
        batch_indices: List[int] = []
        cat5_positions: List[int] = []
        cat5_keys: List[Dict[str, str]] = []

        # ------------------------------------------------------------------
        # Assemble batch
        # ------------------------------------------------------------------
        for i in range(batch_start, min(batch_start + args.batch_size, len(qa_items))):
            if prediction_key in out_data["qa"][i] and not args.overwrite:
                if out_data["qa"][i][prediction_key] != "NO_PREDICTION":
                    continue

            qa = qa_items[i]
            question = qa.get("question", "")
            category = qa.get("category", 0)

            if category == 2:
                question += (
                    " Use DATE of CONVERSATION to answer with an approximate date."
                )
            elif category == 5:
                if "answer" not in qa and "adversarial_answer" not in qa:
                    logger.warning(
                        "Sample %s QA[%d] is category 5 but has neither "
                        "'answer' nor 'adversarial_answer'; skipping.",
                        in_data.get("sample_id", "?"),
                        i,
                    )
                    continue
                distractor = qa.get("answer") or qa.get("adversarial_answer")
                question, key = build_cat5_question(question, distractor)
                cat5_positions.append(len(batch_questions))
                cat5_keys.append(key)

            batch_questions.append(question)
            batch_indices.append(i)

        if not batch_questions:
            continue

        if args.batch_size == 1:
            template = QA_PROMPT_CAT5 if cat5_positions else QA_PROMPT_SINGLE
            question_part = template.format(question=batch_questions[0])
            reserved = count_tokens(question_part) + START_PROMPT_OVERHEAD_TOKENS
        else:
            enumerated = "\n".join(f"{k}: {q}" for k, q in enumerate(batch_questions))

            question_part = QA_PROMPT_BATCH + enumerated

            reserved = count_tokens(question_part) + START_PROMPT_OVERHEAD_TOKENS

        if args.use_rag:
            if args.batch_size == 1:
                question = batch_questions[0]

                retrieved_texts, retrieved_ids, retrieved_dates = retrieve_top_k(
                    question, context_embeddings, texts, ids, dates, top_k=args.top_k
                )

                context = ""
                for r_text, r_id, r_date in zip(
                    retrieved_texts, retrieved_ids, retrieved_dates
                ):
                    context += f"[{r_id} | {r_date}]\n{r_text}\n\n"

                full_query = start_prompt + context + "\n\n" + question_part
            else:
                raise NotImplementedError(
                    "Batch mode with RAG is not implemented. Use batch_size=1."
                )
        else:
            budget = (
                args.context_token_budget if args.access_mode == "truncated" else None
            )
            context = build_context(
                in_data["conversation"], reserved, max_context_tokens=budget
            )
            full_query = start_prompt + context + "\n\n" + question_part

        if "pro-1.0" in args.model:
            import time

            time.sleep(5)

        if args.batch_size == 1:
            idx = batch_indices[0]
            raw = llm_fn(full_query, PER_QA_TOKEN_BUDGET)
            answer = raw.strip()
            if cat5_positions:
                answer = resolve_cat5(answer, cat5_keys[0])
            out_data["qa"][idx][prediction_key] = answer
            continue

        # Batch path this thing is  NOT SUPPORTED WITH RAG

        max_tokens = PER_QA_TOKEN_BUDGET * len(batch_questions)
        raw = llm_fn(full_query, max_tokens)

        parsed: Optional[Dict] = None
        for trial in range(1, 6):
            try:
                parsed = parse_json_output(raw)
                break
            except PARSE_ERRORS as exc:
                logger.warning("Batch parse failed (trial %d/5): %s", trial, exc)
                if trial < 5:
                    raw = llm_fn(full_query, max_tokens)

        if parsed is None:
            logger.error("Abandoning batch after 5 parse attempts.")
            for idx in batch_indices:
                out_data["qa"][idx][prediction_key] = "PARSE_ERROR"
            continue

        for pos, idx in enumerate(batch_indices):
            try:
                ans = parsed[str(pos)]
            except (KeyError, TypeError):
                try:
                    ans = parsed[pos]
                except (KeyError, TypeError):
                    logger.warning(
                        "Missing key '%s' in batch response for question %d", pos, idx
                    )
                    out_data["qa"][idx][prediction_key] = "PARSE_ERROR"
                    continue

            if pos in cat5_positions:
                ans = resolve_cat5(str(ans), cat5_keys[cat5_positions.index(pos)])
            else:
                ans = str(ans).replace("(a)", "").replace("(b)", "").strip()

            out_data["qa"][idx][prediction_key] = ans

    return out_data
