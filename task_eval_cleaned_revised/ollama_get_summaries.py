import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import os, json
import requests
import pickle
from tqdm import tqdm
import argparse
import numpy as np


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = "llama3.2:1b"  # for summarisation
EMBED_MODEL = "nomic-embed-text"  # embeddings


def call_ollama_chat(prompt, max_tokens=256, temperature=1.0):

    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]


def call_ollama_embeddings(texts):
    embeddings = []
    for text in texts:
        payload = {"model": EMBED_MODEL, "prompt": text}
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])
    return np.array(embeddings)


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


def get_session_summary(session, date_time):
    query = get_summary_query(session, date_time)

    session_summary = call_ollama_chat(query, max_tokens=256, temperature=1.0)
    return session_summary


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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Verify Ollama is reachable if not return the erros without crashing the hell out of the script
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
    except Exception as e:
        print(
            f"ERROR: Ollama is not running or not reachable at {OLLAMA_BASE_URL}. Details: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    samples = json.load(open(args.data_file))

    if os.path.exists(args.out_file):
        out_samples = {d["sample_id"]: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:
        summaries = []
        date_times = []
        context_ids = []

        if data["sample_id"] in out_samples:
            output = out_samples["sample_id"]
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
            if "session_%s_summary" % i not in output or args.overwrite:
                summary = get_session_summary(
                    data["conversation"]["session_%s" % i],
                    data["conversation"]["session_%s_date_time" % i],
                )
                output["session_%s_summary" % i] = summary
            else:
                summary = output["session_%s_summary" % i]

            date_time = data["conversation"]["session_%s_date_time" % i]
            summaries.append(summary)
            date_times.append(date_time)
            context_ids.append("S%s" % i)

            print("Getting embeddings for %s summaries" % len(summaries))

            embeddings = call_ollama_embeddings(summaries)

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


# runnnnnnnnn!
if __name__ == "__main__":
    main()
