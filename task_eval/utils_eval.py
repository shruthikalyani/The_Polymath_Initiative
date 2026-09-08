import random
import time

from dotenv import load_dotenv
from groq import Groq
import os
import json
import requests
import tiktoken
from tqdm import tqdm
from global_methods import run_llama_open_router, run_llama_groq

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


def get_llama_answers(in_data, out_data, prediction_key, args):
    assert len(in_data["qa"]) == len(out_data["qa"]), (
        len(in_data["qa"]),
        len(out_data["qa"]),
    )

    # start instruction prompt
    speakers_names = list(
        set([d["speaker"] for d in in_data["conversation"]["session_1"]])
    )
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    # start_tokens = model.count_tokens(start_prompt)
    start_tokens = 100

    if args.rag_mode:
        raise NotImplementedError
    else:
        context_database, query_vectors = None, None

    for batch_start_idx in tqdm(
        range(0, len(in_data["qa"]), args.batch_size), desc="Generating answers"
    ):
        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):
            if i >= len(in_data["qa"]):
                break

            qa = in_data["qa"][i]

            if prediction_key not in out_data["qa"][i] or args.overwrite:
                include_idxs.append(i)
            else:
                continue

            if qa["category"] == 2:
                questions.append(
                    qa["question"]
                    + " Use DATE of CONVERSATION to answer with an approximate date."
                )
            elif qa["category"] == 5:
                question = (
                    qa["question"] + " Select the correct answer: (a) {} (b) {}. "
                )
                if random.random() < 0.5:
                    question = question.format(
                        "Not mentioned in the conversation", qa["answer"]
                    )
                    answer = {
                        "a": "Not mentioned in the conversation",
                        "b": qa["answer"],
                    }
                else:
                    question = question.format(
                        qa["answer"], "Not mentioned in the conversation"
                    )
                    answer = {
                        "b": "Not mentioned in the conversation",
                        "a": qa["answer"],
                    }

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # questions.append(qa['question'] + "Write NOT ANSWERABLE if the question cannot be answered")
            else:
                questions.append(qa["question"])

        if questions == []:
            continue

        context_ids = None
        if args.use_rag:
            raise NotImplementedError
        else:
            question_prompt = QA_PROMPT_BATCH + "\n".join(
                ["%s: %s" % (k, q) for k, q in enumerate(questions)]
            )
            # num_question_tokens = model.count_tokens(question_prompt)
            num_question_tokens = 100
            # Axis B: "truncated" -> real sliding-window truncation against
            # args.context_token_budget; "full" -> no truncation at all
            # (max_context_tokens=None), i.e. the entire conversation goes
            # in every time, however large.
            context_budget = (
                args.context_token_budget if args.access_mode == "truncated" else None
            )
            query_conv = get_input_context(
                in_data["conversation"],
                num_question_tokens + start_tokens,
                None,
                args,
                max_context_tokens=context_budget,
            )
            query_conv = start_prompt + query_conv

        # print("%s tokens in query" % len(model.count_tokens(query_conv)))

        if "pro-1.0" in args.model:
            time.sleep(5)

        if args.batch_size == 1:
            query = (
                query_conv + "\n\n" + QA_PROMPT.format(questions[0])
                if len(cat_5_idxs) == 0
                else query_conv + "\n\n" + QA_PROMPT_CAT_5.format(questions[0])
            )
            answer = LocalGenerator.generate(
                query, PER_QA_TOKEN_BUDGET
            )

            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            out_data["qa"][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data["qa"][include_idxs[0]][prediction_key + "_context"] = (
                    context_ids
                )

        else:
            # query = query_conv + '\n' + QA_PROMPT_BATCH + "\n".join(["QUESTION: %s" % q for q in questions])
            query = query_conv + "\n" + question_prompt
            # print(query)

            trials = 0
            parsed_ok = False
            # Exceptions process_ouput/json.loads can actually raise on malformed
            # model output: JSONDecodeError (bad JSON), ValueError (no "{" found
            # at all), IndexError (empty string after stripping/replacing).
            PARSE_ERRORS = (json.decoder.JSONDecodeError, ValueError, IndexError)
            while trials < 5:
                try:
                    trials += 1
                    # print("Trial %s" % trials)
                    # print("Sending query of %s tokens" % len(model.count_tokens(query)))
                    # print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = LocalGenerator.generate(query, PER_QA_TOKEN_BUDGET * args.batch_size)
                    answer = (
                        answer.replace('\\"', "'")
                        .replace("json", "")
                        .replace("`", "")
                        .strip()
                    )
                    try:
                        answers = json.loads(answer.strip())
                    except Exception as e:
                        print(
                            "~~~~~~~~~~~~ ERROR DURING PARSING REPSONSE AS JSON ~~~~~~~~~~~~~~~"
                        )
                        raise e
                    answers = process_ouput(answer.strip())
                    parsed_ok = True
                    break
                except PARSE_ERRORS as e:
                    print(
                        "~~~~~~~~~~~~ Parse failed on trial %s/%s: %s: %s ~~~~~~~~~~~~"
                        % (trials, 5, type(e).__name__, e)
                    )

            if not parsed_ok:
                # All retries exhausted without valid JSON. Fall back to marking
                # every question in this batch as unparsed instead of silently
                # reusing whatever `answer`/`answers` happened to be left over
                # from the last failed attempt.
                print(
                    "~~~~~~~~~~~~ Giving up after %s trials, could not parse a valid "
                    "JSON response for this batch. Marking as PARSE_ERROR. ~~~~~~~~~~~~"
                    % trials
                )
                for idx in include_idxs:
                    out_data["qa"][idx][prediction_key] = "PARSE_ERROR"
                continue

            for k, idx in enumerate(include_idxs):
                try:
                    answers = process_ouput(answer.strip())
                    # answers = json.loads(answer.strip())
                    # data['qa'][idx]['%s_prediction' % args.model] = answers[k]['answer'].strip()
                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(
                            answers[str(k)], cat_5_answers[cat_5_idxs.index(k)]
                        )
                        out_data["qa"][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data["qa"][idx][prediction_key] = (
                                str(answers[str(k)])
                                .replace("(a)", "")
                                .replace("(b)", "")
                                .strip()
                            )
                        except:
                            out_data["qa"][idx][prediction_key] = ", ".join(
                                [str(n) for n in list(answers[str(k)].values())]
                            )
                except:
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(
                                answers[k], cat_5_answers[cat_5_idxs.index(k)]
                            )
                            out_data["qa"][idx][prediction_key] = predicted_answer
                        else:
                            out_data["qa"][idx][prediction_key] = (
                                answers[k].replace("(a)", "").replace("(b)", "").strip()
                            )
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(
                                answer.strip(), cat_5_answers[cat_5_idxs.index(k)]
                            )
                            out_data["qa"][idx][prediction_key] = predicted_answer
                        else:
                            out_data["qa"][idx][prediction_key] = json.loads(
                                answer.strip()
                                .replace("(a)", "")
                                .replace("(b)", "")
                                .split("\n")[k]
                            )[0]

    return out_data
