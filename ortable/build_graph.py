#!/usr/bin/env python3
"""
build_graph.py — JANUS Symlink Graph Builder
=============================================

Loads trained MLP weights + embedding cache, runs the full corpus through
the MLP, and serializes a pre-computed symlink graph compatible with
janus/symlink_index.py:SymlinkIndex.load().

Run after train_remote.py:
    python build_graph.py

Output:
    data/symlink_index.json   — pre-computed graph (copy to janus/index/)
"""
import json
import math
import random
import hashlib
import datetime
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"
WEIGHTS_PATH = Path(__file__).parent / "link_mlp_trained.json"
CACHE_PATH = DATA_DIR / "embedding_cache.jsonl"
OUTPUT_PATH = DATA_DIR / "symlink_index.json"

TARGET_DIM = 64
LINK_TYPES = ['analogy', 'contrast', 'extension', 'context',
              'resolution', 'metaphor', 'symbol']

LINK_SYMBOLS = {
    'analogy': '→',
    'contrast': '↔',
    'extension': '⇒',
    'context': '≈',
    'resolution': '✓',
    'metaphor': '%%',
    'symbol': '∞',
}

MAX_DISCOVERED_PAIRS = 20000   # sampled non-adjacent pairs to predict
LOSSY_THRESHOLD = 0.5          # links below this confidence are marked lossy


# ─── Minimal MLP forward (no training machinery) ──────────────────

class ForwardMLP:
    """Inference-only MLP. Loads weights from JSON, runs forward pass."""

    def __init__(self, weights_path: Path):
        with open(weights_path) as f:
            d = json.load(f)
        self.input_dim = d['input_dim']
        self.output_labels = d['output_labels']
        # Convert to numpy for speed
        self.weights = [np.array(w, dtype=np.float64) for w in d['weights']]
        self.biases = [np.array(b, dtype=np.float64) for b in d['biases']]

    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        elif len(x) > self.input_dim:
            x = x[:self.input_dim]

        current = x.astype(np.float64)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            current = np.maximum(0.0, z) if i < len(self.weights) - 1 else z

        logits = current[:-1]
        raw_conf = current[-1]
        shifted = logits - logits.max()
        probs = np.exp(shifted) / np.exp(shifted).sum()
        conf = 1.0 / (1.0 + np.exp(-np.clip(raw_conf, -500, 500)))

        label = self.output_labels[int(np.argmax(probs))]
        return label, float(conf)


# ─── Embedding cache (read-only) ─────────────────────────────────

def load_cache(path: Path) -> Dict[str, List[float]]:
    cache = {}
    if not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
                cache[rec['k']] = rec['v']
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def cache_key(text: str) -> str:
    return hashlib.sha256(
        text[:512].encode('utf-8', errors='replace')
    ).hexdigest()[:16]


# ─── Corpus loading ───────────────────────────────────────────────

def extract_top_of_mind(memory_context: str, max_chars: int = 180) -> str:
    if not memory_context:
        return ""
    marker = "**Top of mind**"
    idx = memory_context.find(marker)
    if idx == -1:
        return ""
    rest = memory_context[idx + len(marker):].lstrip('\n ')
    next_header = rest.find("**")
    if next_header != -1:
        rest = rest[:next_header]
    return rest.strip()[:max_chars]


def load_turns(cache: Dict[str, List[float]]) -> List[Dict]:
    """Load corpus turns, resolving embeddings from cache."""
    files = sorted(DATA_DIR.glob("conversations_*.jsonl"))
    turns = []
    missing = 0

    for jf in files:
        with open(jf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                conv_name = rec.get("conversation_name", "").strip()
                top_of_mind = extract_top_of_mind(rec.get("memory_context", ""))

                for t in rec.get("turns", []):
                    text = t.get("text", "").strip()
                    if not text or len(text) <= 20:
                        continue

                    text_trunc = text[:512]
                    parts = []
                    if conv_name:
                        parts.append(f"[{conv_name}]")
                    if top_of_mind:
                        parts.append(f"[{top_of_mind}]")
                    parts.append(text_trunc)
                    embed_key_text = " ".join(parts)

                    k = cache_key(embed_key_text)
                    embedding = cache.get(k)

                    if embedding is None:
                        # Fall back to plain text key (old cache format)
                        k_plain = cache_key(text_trunc)
                        embedding = cache.get(k_plain)

                    if embedding is None:
                        missing += 1
                        continue

                    # Stable chunk ID: sha256 of text (matches SymlinkIndex)
                    chunk_id = hashlib.sha256(
                        text_trunc.encode()
                    ).hexdigest()[:12]

                    turns.append({
                        "chunk_id": chunk_id,
                        "text": text_trunc,
                        "sender": t.get("sender", "?"),
                        "conv": conv_name,
                        "turn_id": t.get("turn_id", ""),
                        "embedding": embedding,
                    })

    if missing:
        print(f"  WARNING: {missing} turns had no cached embedding (skipped)")
    return turns


# ─── Cosine similarity ────────────────────────────────────────────

def cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    a_, b_ = np.array(a), np.array(b)
    na, nb = np.linalg.norm(a_), np.linalg.norm(b_)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_, b_) / (na * nb))


# ─── Graph builder ────────────────────────────────────────────────

def build_graph(turns: List[Dict], mlp: ForwardMLP,
                max_discovered: int = MAX_DISCOVERED_PAIRS
                ) -> Dict[str, Dict]:
    """
    Build the symlink graph.

    Returns chunks dict in SymlinkIndex.save() format:
    { chunk_id: { text, links: [...], schema_tags: {} } }
    """
    chunks: Dict[str, Dict] = {}
    for t in turns:
        chunks[t["chunk_id"]] = {
            "text": t["text"],
            "links": [],
            "schema_tags": {
                "sender": t["sender"],
                "conv": t["conv"],
                "turn_id": t["turn_id"],
            },
        }

    link_counts = Counter()
    total_links = 0

    def add_link(src_id: str, tgt_id: str, link_type: str, confidence: float):
        nonlocal total_links
        symbol = LINK_SYMBOLS.get(link_type, '→')
        lossy = confidence < LOSSY_THRESHOLD
        chunks[src_id]["links"].append({
            "target": tgt_id,
            "type": link_type,
            "value": round(confidence, 4),
            "symbols": symbol,
            "confidence": round(confidence, 4),
            "lossy": lossy,
            "hyper_depth": 1,
        })
        link_counts[link_type] += 1
        total_links += 1

    # Phase 1: positional pairs (adjacent turns within same conversation)
    print(f"\nPhase 1: positional pairs...")
    positional = 0
    for i in range(len(turns) - 1):
        t1, t2 = turns[i], turns[i + 1]
        if t1["conv"] != t2["conv"]:
            continue

        x = np.array(t1["embedding"] + t2["embedding"], dtype=np.float64)
        link_type, confidence = mlp.predict(x)
        add_link(t1["chunk_id"], t2["chunk_id"], link_type, confidence)
        positional += 1

    print(f"  {positional} positional links")

    # Phase 2: sampled non-adjacent pairs
    print(f"Phase 2: discovered pairs (sampling {max_discovered})...")
    n = len(turns)
    sampled = set()
    attempts = 0
    target = min(max_discovered, n * (n - 1) // 2)

    while len(sampled) < target and attempts < target * 4:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if (i != j and abs(i - j) > 1
                and (i, j) not in sampled and (j, i) not in sampled):
            sampled.add((i, j))
        attempts += 1

    discovered = 0
    for i, j in sampled:
        t1, t2 = turns[i], turns[j]
        x = np.array(t1["embedding"] + t2["embedding"], dtype=np.float64)
        link_type, confidence = mlp.predict(x)
        add_link(t1["chunk_id"], t2["chunk_id"], link_type, confidence)
        discovered += 1

        if discovered % 5000 == 0:
            print(f"  {discovered}/{len(sampled)}...")

    print(f"  {discovered} discovered links")
    print(f"\n  total links: {total_links}")
    print(f"  link types: {dict(link_counts)}")

    return chunks


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  JANUS Symlink Graph Builder")
    print("=" * 64)

    if not WEIGHTS_PATH.exists():
        print(f"ERROR: {WEIGHTS_PATH} not found. Run train_remote.py first.")
        return

    print(f"\nLoading MLP weights from {WEIGHTS_PATH.name}...")
    mlp = ForwardMLP(WEIGHTS_PATH)
    print(f"  input_dim={mlp.input_dim}  labels={mlp.output_labels}")

    print(f"\nLoading embedding cache...")
    cache = load_cache(CACHE_PATH)
    print(f"  {len(cache)} vectors")

    print(f"\nLoading corpus turns...")
    turns = load_turns(cache)
    print(f"  {len(turns)} turns with embeddings")

    if len(turns) < 4:
        print("Not enough turns.")
        return

    # Deduplicate by chunk_id (same text → same id)
    seen = {}
    deduped = []
    for t in turns:
        if t["chunk_id"] not in seen:
            seen[t["chunk_id"]] = True
            deduped.append(t)
    if len(deduped) < len(turns):
        print(f"  deduped {len(turns) - len(deduped)} duplicate turns → {len(deduped)} unique")
    turns = deduped

    # Build graph
    chunks = build_graph(turns, mlp)

    # Serialize in SymlinkIndex.save() format
    data = {
        "chunks": chunks,
        "schema_dimensions": {
            "geometry": ["linear", "recursive", "spiral", "bifurcating"],
            "coherence": ["tight", "loose", "fragmented", "emergent"],
            "texture": ["dense", "sparse", "lyrical", "technical"],
            "terrain": ["conceptual", "emotional", "procedural", "speculative"],
        },
        "hyper_symbolic": {
            "use_mlp": True,
            "hyper_depth": 1,
            "lossy_threshold": LOSSY_THRESHOLD,
        },
        "timestamp": datetime.datetime.now().isoformat(),
        "meta": {
            "turns": len(turns),
            "mlp_weights": str(WEIGHTS_PATH),
            "backend": "voyage",
            "embedding_dim": TARGET_DIM,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting graph → {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(data, f)

    size_mb = OUTPUT_PATH.stat().st_size / 1_000_000
    print(f"  {size_mb:.1f} MB  ({len(chunks)} chunks)")
    print(f"\nNext: copy to janus/index/symlink_index.json")
    print("=" * 64)


if __name__ == "__main__":
    main()
