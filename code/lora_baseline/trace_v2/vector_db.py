"""Per-user vector DB — emulates what the LoRA stores, retrieved via cosine similarity.

Vector search is the external analogue of in-context attention:
softmax(QK^T / sqrt(d)) V ~~ topk(cos(q, k_i)). Forcing the agent to look up
facts through this tool prevents it from leaning on in-context fact dumps
during trace generation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


REPO = Path(__file__).resolve().parents[2]
USERS_DIR = REPO / "data" / "users"
VECTOR_CACHE_DIR = REPO / "data" / "trace_v2" / "vector_cache"
EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"


@dataclass
class RecallHit:
    fact_key: str
    fact_text: str
    answer: object
    score: float

    def to_dict(self) -> dict:
        return {
            "fact_key": self.fact_key,
            "fact_text": self.fact_text,
            "score": round(float(self.score), 4),
        }


def _canonical_fact_text(fact: dict) -> str:
    """One-line canonical rendering of a fact for display to the agent."""
    paraphrases = fact.get("paraphrases") or []
    if paraphrases:
        return paraphrases[0]
    return f"{fact['key']}: {fact['answer']}"


@lru_cache(maxsize=1)
def _embed_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_ID)


@dataclass
class UserIndex:
    uid: str
    # parallel arrays, one entry per (fact, paraphrase) document
    doc_embeddings: np.ndarray  # (N, D), L2-normalized
    doc_fact_keys: list[str]
    fact_key_to_text: dict[str, str]
    fact_key_to_answer: dict[str, object]

    @property
    def n_docs(self) -> int:
        return len(self.doc_fact_keys)


def _load_user(uid: str) -> dict:
    return json.loads((USERS_DIR / f"{uid}.json").read_text())


def _build_index(uid: str) -> UserIndex:
    user = _load_user(uid)
    docs: list[str] = []
    fact_keys: list[str] = []
    key_to_text: dict[str, str] = {}
    key_to_answer: dict[str, object] = {}

    for fact in user["facts"]:
        key = fact["key"]
        key_to_text[key] = _canonical_fact_text(fact)
        key_to_answer[key] = fact["answer"]
        paraphrases = fact.get("paraphrases") or [key_to_text[key]]
        for p in paraphrases:
            docs.append(p)
            fact_keys.append(key)

    model = _embed_model()
    # bge-small expects a "query:" prefix for queries; docs are plain.
    emb = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return UserIndex(
        uid=uid,
        doc_embeddings=emb.astype(np.float32),
        doc_fact_keys=fact_keys,
        fact_key_to_text=key_to_text,
        fact_key_to_answer=key_to_answer,
    )


def _cache_path(uid: str) -> Path:
    return VECTOR_CACHE_DIR / f"{uid}.npz"


def _save_index(idx: UserIndex) -> None:
    VECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _cache_path(idx.uid),
        emb=idx.doc_embeddings,
        fact_keys=np.array(idx.doc_fact_keys, dtype=object),
        keys=np.array(list(idx.fact_key_to_text.keys()), dtype=object),
        texts=np.array(list(idx.fact_key_to_text.values()), dtype=object),
        answers=np.array(
            [json.dumps(idx.fact_key_to_answer[k]) for k in idx.fact_key_to_text.keys()],
            dtype=object,
        ),
    )


def _load_cached_index(uid: str) -> UserIndex | None:
    p = _cache_path(uid)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    keys = list(z["keys"])
    texts = list(z["texts"])
    answers = [json.loads(a) for a in z["answers"]]
    return UserIndex(
        uid=uid,
        doc_embeddings=z["emb"].astype(np.float32),
        doc_fact_keys=list(z["fact_keys"]),
        fact_key_to_text=dict(zip(keys, texts)),
        fact_key_to_answer=dict(zip(keys, answers)),
    )


@lru_cache(maxsize=64)
def get_index(uid: str) -> UserIndex:
    cached = _load_cached_index(uid)
    if cached is not None:
        return cached
    idx = _build_index(uid)
    _save_index(idx)
    return idx


def recall(uid: str, query: str, top_k: int = 3) -> list[RecallHit]:
    """Retrieve top-k facts for `uid` most similar to `query`, deduped by fact_key.

    Returned list is length <= top_k; scores are cosine similarities in [-1, 1].
    """
    idx = get_index(uid)
    model = _embed_model()
    q = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    sims = idx.doc_embeddings @ q  # (N,)

    # dedupe by fact_key: keep best-scoring doc per key
    best_per_key: dict[str, float] = {}
    for score, key in zip(sims, idx.doc_fact_keys):
        s = float(score)
        if key not in best_per_key or s > best_per_key[key]:
            best_per_key[key] = s

    ranked = sorted(best_per_key.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RecallHit(
            fact_key=k,
            fact_text=idx.fact_key_to_text[k],
            answer=idx.fact_key_to_answer[k],
            score=s,
        )
        for k, s in ranked
    ]


def warm_all(uids: list[str]) -> None:
    for uid in uids:
        get_index(uid)


if __name__ == "__main__":
    import sys

    uid = sys.argv[1] if len(sys.argv) > 1 else "u000"
    queries = [
        "how old am I",
        "my spouse's age",
        "what days do I go to the gym",
        "who is my emergency contact",
        "do I commute by car",
    ]
    for q in queries:
        hits = recall(uid, q, top_k=3)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h.score:.3f}] {h.fact_key}: {h.fact_text}")
