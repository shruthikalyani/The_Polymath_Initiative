"""LoCoMo memory-architecture adapters for the Poly Math Initiative study.

Implements the six planned conditions:
1. raw_truncated
2. raw_full
3. summary_rag
4. facts_rag
5. summary_tiered
6. graph_traversal

This module deliberately keeps LoCoMo's conversation/QA data format and exposes
one common `build_memory` + `retrieve_context` interface.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from global_methods import run_llama_groq


EXPERIMENTS = {
    "raw_truncated": {"storage": "raw", "access": "truncated", "label": "1_raw_truncated"},
    "raw_full": {"storage": "raw", "access": "full", "label": "2_raw_full"},
    "summary_rag": {"storage": "summaries", "access": "vector", "label": "3_summary_rag"},
    "facts_rag": {"storage": "facts", "access": "vector", "label": "4_facts_rag"},
    "summary_tiered": {"storage": "summaries+raw", "access": "tiered", "label": "5_summary_tiered"},
    "graph_traversal": {"storage": "graph", "access": "graph", "label": "6_graph_traversal"},
}

# Original LoCoMo session-summary instruction from get_session_summaries.get_summary_query.
SUMMARY_PROMPT = (
    "Generate a concise summary of the following conversation using exact words "
    "from the conversation wherever possible. The summary should contain all facts "
    "about the two speakers, as well as references to time.\n"
    "{conversation}\n"
)

FACTS_PROMPT = """
Extract objective, speaker-specific factual memories from the following LoCoMo conversation.
Return a JSON object with exactly two keys, one per speaker. Each value must be a list of
objects with fields: "fact" and "dialog_ids". Include only information explicitly supported
by the conversation. Keep facts concise. Do not output opinions about conversational dynamics.

DATE: {date}
CONVERSATION:
{conversation}
""".strip()

GRAPH_PROMPT = """
Extract a small factual knowledge graph from this conversation.
Return JSON with keys:
- "nodes": a list of objects {"id": string, "label": string, "type": string}
- "edges": a list of objects {"source": string, "relation": string, "target": string, "dialog_ids": [string]}
Use only explicit facts from the conversation. Do not invent entities or relations.
Keep node labels canonical and concise.

DATE: {date}
CONVERSATION:
{conversation}
""".strip()


def _clean_json_text(text: str) -> str:
    text = (text or "").strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("No JSON object/array found in model output")
    return text[min(starts):]


def _call_llm(prompt: str, max_tokens: int, llm_fn=None) -> str:
    fn = llm_fn
    return fn(prompt, max_tokens)


def _llm_json(prompt: str, max_tokens: int = 700, llm_fn=None) -> Any:
    last = None
    for _ in range(5):
        try:
            out = _call_llm(prompt, max_tokens, llm_fn=llm_fn)
            return json.loads(_clean_json_text(out))
        except Exception as exc:
            last = exc
    raise ValueError(f"Could not parse model JSON after 5 attempts: {last}")


def _conversation_sessions(conversation: Dict[str, Any]) -> List[Tuple[int, List[Dict[str, Any]], str]]:
    found = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and isinstance(value, list):
            idx = int(key.split("_")[1])
            found.append((idx, value, conversation.get(f"session_{idx}_date_time", "")))
    found.sort(key=lambda x: x[0])
    return found


def _turn_text(turn: Dict[str, Any]) -> str:
    text = turn.get("clean_text", turn.get("text", ""))
    return f'{turn.get("speaker", "Unknown")}: {text}'


def _session_text(turns: Sequence[Dict[str, Any]]) -> str:
    rows = []
    for t in turns:
        row = _turn_text(t)
        if t.get("blip_caption"):
            row += f' [shared: {t["blip_caption"]}]'
        rows.append(row)
    return "\n".join(rows)


def _summary_conversation_block(turns: Sequence[Dict[str, Any]], date: str) -> str:
    """Match LoCoMo get_summary_query formatting (date + speaker said, \"text\")."""
    conv = f"{date}\n"
    for dialog in turns:
        conv += dialog.get("speaker", "Unknown") + ' said, "' + dialog.get("text", dialog.get("clean_text", "")) + '"'
        if dialog.get("blip_caption"):
            conv += "and shared " + dialog["blip_caption"] + "."
        conv += "\n"
    return conv


def render_turns_with_dates(turns: Sequence[Dict[str, Any]]) -> str:
    lines = []
    last_session = None
    for t in turns:
        idx = t.get("session_idx")
        if idx != last_session:
            date = t.get("date", "")
            lines.append(f"DATE: {date}")
            lines.append("CONVERSATION:")
            last_session = idx
        lines.append(f'{t.get("speaker", "Unknown")}: {t.get("text", "")}')
    return "\n".join(lines)


def flatten_turns(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for idx, turns, date in _conversation_sessions(conversation):
        for turn in turns:
            out.append({
                "session_idx": idx,
                "date": date,
                "speaker": turn.get("speaker", "Unknown"),
                "text": turn.get("clean_text", turn.get("text", "")),
                "dia_id": turn.get("dia_id"),
            })
    return out


def token_count(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def sliding_window_context(
    conversation: Dict[str, Any], budget_tokens: int, tokenizer=None
) -> Dict[str, Any]:
    """True turn-level newest-first sliding window, not session-level truncation."""
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    turns = flatten_turns(conversation)
    kept = []
    used = 0
    for turn in reversed(turns):
        n = token_count(render_turns_with_dates([turn]), tokenizer)
        if kept and used + n > budget_tokens:
            break
        kept.append(turn)
        used += n
    kept.reverse()
    text = render_turns_with_dates(kept)
    return {
        "turns": kept,
        "token_count": token_count(text, tokenizer),
        "text": text,
    }


class VectorStore:
    """Minimal cosine-similarity store. Uses the retriever already selected by the project."""

    def __init__(self, retriever: str = "contriever", device: Optional[str] = None):
        self.retriever = retriever
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self._context_tokenizer = None
        self._context_encoder = None
        self._query_tokenizer = None
        self._query_encoder = None
        self.matrix: Optional[np.ndarray] = None
        self.items: List[Dict[str, Any]] = []

    def _lazy_init(self):
        if self.retriever == "openai":
            # LoCoMo's old OpenAI embedding path is intentionally not used here.
            raise ValueError("Use a local Hugging Face retriever (contriever/dpr/dragon) for this study")
        from transformers import AutoTokenizer, AutoModel

        if self._context_encoder is None:
            if self.retriever == "contriever":
                self._context_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
                self._context_encoder = AutoModel.from_pretrained("facebook/contriever").to(self.device).eval()
                self._query_tokenizer = self._context_tokenizer
                self._query_encoder = self._context_encoder
            elif self.retriever == "dragon":
                self._context_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
                self._context_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder").to(self.device).eval()
                self._query_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
                self._query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder").to(self.device).eval()
            else:
                raise ValueError(f"Unsupported local retriever: {self.retriever}")

    @staticmethod
    def _mean_pool(hidden, mask):
        mask = mask.unsqueeze(-1).expand(hidden.size()).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def _encode(self, texts: Sequence[str], query: bool = False, batch_size: int = 24) -> np.ndarray:
        self._lazy_init()
        tokenizer = self._query_tokenizer if query else self._context_tokenizer
        encoder = self._query_encoder if query else self._context_encoder
        rows = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = encoder(**toks).last_hidden_state
                emb = self._mean_pool(out, toks["attention_mask"])
                emb = torch.nn.functional.normalize(emb, dim=-1)
            rows.append(emb.cpu().numpy())
        return np.concatenate(rows, axis=0) if rows else np.empty((0, 768), dtype=np.float32)

    def fit(self, items: List[Dict[str, Any]]):
        self.items = items
        if not items:
            self.matrix = np.empty((0, 768), dtype=np.float32)
            return self
        self.matrix = self._encode([x["text"] for x in items], query=False)
        return self

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.matrix is None or not self.items:
            return []
        q = self._encode([query], query=True)[0]
        sims = self.matrix @ q
        idxs = np.argsort(sims)[::-1][: min(top_k, len(self.items))]
        result = []
        for i in idxs:
            obj = dict(self.items[int(i)])
            obj["score"] = float(sims[int(i)])
            result.append(obj)
        return result

    def save(self, path: str):
        payload = {"retriever": self.retriever, "items": self.items, "matrix": self.matrix.tolist() if self.matrix is not None else None}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(payload["retriever"], device=device)
        obj.items = payload["items"]
        obj.matrix = np.asarray(payload["matrix"], dtype=np.float32)
        return obj


def build_summaries(conversation: Dict[str, Any], model: str, api_key: str, llm_fn=None) -> List[Dict[str, Any]]:
    items = []
    for idx, turns, date in _conversation_sessions(conversation):
        prompt = SUMMARY_PROMPT.format(conversation=_summary_conversation_block(turns, date))
        summary = _call_llm(prompt, 256, llm_fn=llm_fn).strip()
        items.append({"session_idx": idx, "date": date, "text": summary, "source": f"session_{idx}"})
    return items


def build_facts(conversation: Dict[str, Any], llm_fn=None) -> List[Dict[str, Any]]:
    items = []
    for idx, turns, date in _conversation_sessions(conversation):
        data = _llm_json(
            FACTS_PROMPT.format(date=date, conversation=_session_text(turns)),
            700, llm_fn=llm_fn,
        )
        for speaker, facts in data.items():
            if not isinstance(facts, list):
                continue
            for item in facts:
                if isinstance(item, str):
                    fact = item
                    ids = []
                else:
                    fact = str(item.get("fact", "")).strip()
                    ids = item.get("dialog_ids", []) or []
                if fact:
                    items.append({
                        "session_idx": idx,
                        "date": date,
                        "speaker": speaker,
                        "text": fact,
                        "dialog_ids": ids,
                        "source": f"session_{idx}",
                    })
    return items


def build_graph(conversation: Dict[str, Any], llm_fn=None) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    for idx, turns, date in _conversation_sessions(conversation):
        data = _llm_json(
            GRAPH_PROMPT.format(date=date, conversation=_session_text(turns)),
            900, llm_fn=llm_fn,
        )
        for node in data.get("nodes", []) if isinstance(data, dict) else []:
            nid = str(node.get("id", "")).strip()
            if nid:
                nodes[nid] = {
                    "id": nid,
                    "label": str(node.get("label", nid)).strip(),
                    "type": str(node.get("type", "entity")).strip(),
                    "session_idx": idx,
                    "date": date,
                }
        for edge in data.get("edges", []) if isinstance(data, dict) else []:
            s = str(edge.get("source", "")).strip()
            t = str(edge.get("target", "")).strip()
            r = str(edge.get("relation", "")).strip()
            if s and t and r:
                edges.append({
                    "source": s,
                    "relation": r,
                    "target": t,
                    "dialog_ids": edge.get("dialog_ids", []) or [],
                    "session_idx": idx,
                    "date": date,
                })
    return {"nodes": list(nodes.values()), "edges": edges}


def _norm(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def graph_retrieve(graph: Dict[str, Any], question: str, max_hops: int = 1, top_k_edges: int = 8) -> List[str]:
    q = _norm(question)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    scored = []
    for n in nodes:
        overlap = len(q & _norm(n.get("label", "")))
        if overlap:
            scored.append((overlap, n["id"]))
    scored.sort(reverse=True)
    seeds = {nid for _, nid in scored[:8]}
    if not seeds:
        # Fall back to relation/edge lexical overlap when entity labels have no exact token match.
        edge_hits = []
        for e in edges:
            score = len(q & (_norm(e["relation"]) | _norm(e["source"]) | _norm(e["target"])))
            if score:
                edge_hits.append((score, e))
        edge_hits.sort(key=lambda x: x[0], reverse=True)
        return [f'{e["source"]} --{e["relation"]}--> {e["target"]}' for _, e in edge_hits[:top_k_edges]]

    frontier = set(seeds)
    visited = set(seeds)
    chosen = []
    for _ in range(max_hops):
        nxt = set()
        for e in edges:
            if e["source"] in frontier or e["target"] in frontier:
                chosen.append(e)
                other = e["target"] if e["source"] in frontier else e["source"]
                if other not in visited:
                    nxt.add(other)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    # Keep deterministic order and cap.
    dedup = []
    seen = set()
    for e in chosen:
        key = (e["source"], e["relation"], e["target"])
        if key not in seen:
            seen.add(key)
            dedup.append(f'{e["source"]} --{e["relation"]}--> {e["target"]}')
    return dedup[:top_k_edges]


@dataclass
class MemoryBundle:
    experiment: str
    storage: str
    access: str
    artifacts: Dict[str, Any]
    metadata: Dict[str, Any]


def build_memory(
    conversation: Dict[str, Any],
    experiment: str,
    retriever: str = "contriever",
    artifact_dir: Optional[str] = None,
    llm_fn=None,    
) -> MemoryBundle:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    spec = EXPERIMENTS[experiment]
    artifacts: Dict[str, Any] = {}
    metadata = {
        "experiment": experiment,
        "model": model,
        "provider": provider,
        "retriever": retriever,
        "cache_key": cache_key,
        "memory_format_version": 1,
        **spec,
    }

    if experiment in {"raw_truncated", "raw_full"}:
        return MemoryBundle(experiment, spec["storage"], spec["access"], artifacts, metadata)

    if experiment == "summary_rag":
        items = build_summaries(conversation, model, api_key, llm_fn=llm_fn)
        store = VectorStore(retriever).fit(items)
        artifacts["items"] = items
        artifacts["store"] = store
    elif experiment == "facts_rag":
        items = build_facts(conversation, model, api_key, llm_fn=llm_fn)
        store = VectorStore(retriever).fit(items)
        artifacts["items"] = items
        artifacts["store"] = store
    elif experiment == "summary_tiered":
        items = build_summaries(conversation, model, api_key, llm_fn=llm_fn)
        store = VectorStore(retriever).fit(items)
        artifacts["summary_items"] = items
        artifacts["summary_store"] = store
        artifacts["raw_turns"] = flatten_turns(conversation)
    elif experiment == "graph_traversal":
        artifacts["graph"] = build_graph(conversation, model, api_key, llm_fn=llm_fn)
    return MemoryBundle(experiment, spec["storage"], spec["access"], artifacts, metadata)


def _fit_turns(turns: List[Dict[str, Any]], budget: int, tokenizer, prefix: str, suffix: str) -> List[Dict[str, Any]]:
    kept = []
    for turn in reversed(turns):
        candidate = list(reversed(kept + [turn]))
        text = "\n".join(
            x for x in [prefix, render_turns_with_dates(candidate) if candidate else "", suffix] if x
        )
        if kept and token_count(text, tokenizer) > budget:
            break
        kept.append(turn)
    kept.reverse()
    return kept


def retrieve_context(
    bundle: MemoryBundle,
    conversation: Dict[str, Any],
    question: str,
    tokenizer=None,
    context_budget: int = 6000,
    top_k: int = 5,
    tier_working_turns: int = 12,
) -> Tuple[str, Dict[str, Any]]:
    exp = bundle.experiment
    if exp == "raw_truncated":
        obj = sliding_window_context(conversation, context_budget, tokenizer)
        return obj["text"], {
            "accessed": obj["turns"],
            "token_count": obj["token_count"],
            "truncated_to_budget": True,
        }
    if exp == "raw_full":
        turns = flatten_turns(conversation)
        text = render_turns_with_dates(turns)
        n = token_count(text, tokenizer)
        if n > context_budget:
            obj = sliding_window_context(conversation, context_budget, tokenizer)
            return obj["text"], {
                "accessed": obj["turns"],
                "token_count": obj["token_count"],
                "truncated_to_budget": True,
                "full_token_count": n,
            }
        return text, {
            "accessed": "all_raw_turns",
            "token_count": n,
            "truncated_to_budget": False,
        }
    if exp in {"summary_rag", "facts_rag"}:
        hits = bundle.artifacts["store"].search(question, top_k=top_k)
        chunks = []
        for h in hits:
            prefix = f'[{h.get("date", "")}]'
            speaker = f' {h["speaker"]}:' if h.get("speaker") else ''
            chunks.append(f'{prefix}{speaker} {h["text"]}')
        text = "\n".join(chunks)
        return text, {"retrieved": hits, "token_count": token_count(text, tokenizer)}
    if exp == "summary_tiered":
        raw = bundle.artifacts["raw_turns"]
        working = raw[-tier_working_turns:]
        summaries = bundle.artifacts["summary_store"].search(question, top_k=top_k)
        recalled_sessions = {s["session_idx"] for s in summaries}
        working_ids = {(t.get("dia_id"), t["session_idx"], t["text"]) for t in working}
        paged = [
            t for t in raw
            if t["session_idx"] in recalled_sessions
            and (t.get("dia_id"), t["session_idx"], t["text"]) not in working_ids
        ]
        chunks = ["[WORKING RAW]"]
        chunks.append(render_turns_with_dates(working) if working else "")
        if paged:
            chunks.append("[PAGED RAW FROM RECALLED SESSIONS]")
            chunks.append(render_turns_with_dates(paged))
        chunks += [f'[RECALL SUMMARY {s.get("date", "")}] {s["text"]}' for s in summaries]
        text = "\n".join(c for c in chunks if c)
        truncated = False
        if token_count(text, tokenizer) > context_budget:
            prefix = "[WORKING RAW]\n" + (render_turns_with_dates(working) if working else "")
            suffix = "\n".join(f'[RECALL SUMMARY {s.get("date", "")}] {s["text"]}' for s in summaries)
            paged = _fit_turns(paged, context_budget, tokenizer, prefix, suffix)
            chunks = ["[WORKING RAW]", render_turns_with_dates(working)]
            if paged:
                chunks.append("[PAGED RAW FROM RECALLED SESSIONS]")
                chunks.append(render_turns_with_dates(paged))
            chunks += [f'[RECALL SUMMARY {s.get("date", "")}] {s["text"]}' for s in summaries]
            text = "\n".join(c for c in chunks if c)
            truncated = True
        return text, {
            "working_set": working,
            "paged_sessions": sorted(recalled_sessions),
            "paged_turns": paged,
            "recalled": summaries,
            "token_count": token_count(text, tokenizer),
            "truncated_to_budget": truncated,
        }
    if exp == "graph_traversal":
        facts = graph_retrieve(bundle.artifacts["graph"], question)
        text = "\n".join(f'[GRAPH] {x}' for x in facts)
        return text, {"graph_facts": facts, "token_count": token_count(text, tokenizer)}
    raise ValueError(exp)


def persist_bundle(
    bundle: MemoryBundle, directory: str, sample_id: str, cache_key: Optional[str] = None
):
    """Persist generated memory artifacts so repeated QA runs do not regenerate memory."""
    cache_key = cache_key or bundle.metadata.get("cache_key")
    if not cache_key:
        raise ValueError("A cache_key is required to persist model-generated memory artifacts.")
    out_dir = Path(directory) / cache_key / sample_id / bundle.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2)

    if bundle.experiment == "summary_rag":
        bundle.artifacts["store"].save(str(out_dir / "vector_store.json"))
        with open(out_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(bundle.artifacts["items"], f, indent=2)
    elif bundle.experiment == "facts_rag":
        bundle.artifacts["store"].save(str(out_dir / "vector_store.json"))
        with open(out_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(bundle.artifacts["items"], f, indent=2)
    elif bundle.experiment == "summary_tiered":
        bundle.artifacts["summary_store"].save(str(out_dir / "summary_vector_store.json"))
        with open(out_dir / "summary_items.json", "w", encoding="utf-8") as f:
            json.dump(bundle.artifacts["summary_items"], f, indent=2)
    elif bundle.experiment == "graph_traversal":
        with open(out_dir / "graph.json", "w", encoding="utf-8") as f:
            json.dump(bundle.artifacts["graph"], f, indent=2)
