import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import argparse
import os, json
from generative_agents.memory_utils import get_session_facts
from global_methods import run_llama_open_router, run_llama_groq
from task_eval.rag_utils import get_embeddings
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--emb-dir", type=str, default="")
    parser.add_argument("--prompt-dir", type=str, default="")
    parser.add_argument("--use-date", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="set flag to overwrite existing outputs",
    )
    parser.add_argument("--retriever", type=str, default="dragon")
    parser.add_argument("--provider", type=str, choices=["openrouter", "groq"], default="openrouter")
    parser.add_argument("--model", type=str, default="thinkingmachines/inkling:free")

    args = parser.parse_args()
    return args


def get_summary_query(session, date_time):
    conv = ""
    conv = conv + date_time + "\n"
    for dialog in session:
        conv = conv + dialog["speaker"] + ' said, "' + dialog["text"] + '"'
        if "blip_caption" in dialog:
            conv += "and shared " + dialog["blip_caption"] + "."
        conv = conv + "\n"

    query = "Generate a concise summary of the following conversation using exact words from the conversation wherever possible. The summary should contain all facts about the two speakers, as well as references to time.\n"
    query = query + conv + "\n"
    return query


def get_session_summary(session, date_time, model, provider="openrouter"):
    query = get_summary_query(session, date_time)
    llm = run_llama_open_router if provider == "openrouter" else run_llama_groq
    session_summary = llm(query, 256, model=model)
    return session_summary


def main():
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:
        summaries = []
        date_times = []
        context_ids = []

        # check for existing output
        if data["sample_id"] in out_samples:
            output = out_samples[data["sample_id"]]
        else:
            output = {"sample_id": data["sample_id"]}

        session_nums = [
            int(k.split("_")[-1])
            for k in data["conversation"].keys()
            if "session" in k and "date_time" not in k
        ]
        for i in tqdm(
            range(min(session_nums), max(session_nums) + 1),
            desc="Generating summaries for %s" % data["sample_id"],
        ):
            # get the summaries
            if "session_%s_summary" % i not in output or args.overwrite:
                summary = get_session_summary(
                    data["conversation"]["session_%s" % i],
                    data["conversation"]["session_%s_date_time" % i],
                    args.model,
                    args.provider,
                )
                output["session_%s_summary" % i] = summary
            else:
                summary = output["session_%s_summary" % i]

            date_time = data["conversation"]["session_%s_date_time" % i]
            summaries.append(summary)
            date_times.append(date_time)
            context_ids.append("S%s" % i)

            print("Getting embeddings for %s summaries" % len(summaries))
            embeddings = get_embeddings(args.retriever, summaries, "context")
            assert embeddings.shape[0] == len(summaries)
            database = {
                "embeddings": embeddings,
                "date_time": date_times,
                "dia_id": context_ids,
                "context": summaries,
            }

        with open(
            args.out_file.replace(".json", "_%s.pkl" % data["sample_id"]), "wb"
        ) as f:
            pickle.dump(database, f)

        out_samples[output["sample_id"]] = output.copy()

    with open(args.out_file, "w") as f:
        json.dump(list(out_samples.values()), f, indent=2)


main()
