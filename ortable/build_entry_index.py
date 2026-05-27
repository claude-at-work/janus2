#!/usr/bin/env python3
"""
build_entry_index.py — Semantic entry point index for the symlink graph
=======================================================================

Embeds all graph nodes with all-MiniLM-L6-v2 (same model as Ollama
all-minilm:33m), builds a numpy cosine-search matrix, and writes:

    data/entry_embeddings.npy    — (N, 64) float32 matrix
    data/entry_chunk_ids.json    — [chunk_id, ...] in row order

At query time, memory.py embeds the query with the same model,
does cosine search, and gets proper semantic entry points into the graph.

Run after build_graph.py:
    python build_entry_index.py
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
GRAPH_PATH   = DATA_DIR / "symlink_index.json"
EMB_PATH     = DATA_DIR / "entry_embeddings.npy"
IDS_PATH     = DATA_DIR / "entry_chunk_ids.json"

TARGET_DIM = 64
ST_MODEL   = "all-MiniLM-L6-v2"


def reduce_dim(vec, target=TARGET_DIM):
    if len(vec) <= target:
        return list(vec) + [0.0] * (target - len(vec))
    step = len(vec) / target
    return [float(vec[int(i * step)]) for i in range(target)]


def main():
    print("=" * 64)
    print("  JANUS Entry Index Builder")
    print(f"  model: {ST_MODEL}  →  {TARGET_DIM}-dim")
    print("=" * 64)

    if not GRAPH_PATH.exists():
        print(f"ERROR: {GRAPH_PATH} not found. Run build_graph.py first.")
        return

    print(f"\nLoading graph...")
    with open(GRAPH_PATH) as f:
        data = json.load(f)
    chunks = data.get("chunks", {})
    print(f"  {len(chunks)} chunks")

    chunk_ids = list(chunks.keys())
    texts     = [chunks[cid]["text"][:512] for cid in chunk_ids]

    print(f"\nLoading {ST_MODEL}...")
    from sentence_transformers import SentenceTransformer
    import torch
    model = SentenceTransformer(ST_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  device: {device}")

    print(f"\nEmbedding {len(texts)} texts (batch_size=128)...")
    vecs = model.encode(texts, batch_size=128, show_progress_bar=True)
    print(f"  raw dim: {vecs.shape[1]}")

    print(f"\nStride-sampling to {TARGET_DIM}-dim...")
    reduced = np.array([reduce_dim(v) for v in vecs], dtype=np.float32)
    print(f"  matrix shape: {reduced.shape}")

    # L2-normalise for fast cosine via dot product
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalised = (reduced / norms).astype(np.float32)

    print(f"\nWriting {EMB_PATH.name}  ({normalised.nbytes / 1e6:.1f} MB)...")
    np.save(str(EMB_PATH), normalised)

    print(f"Writing {IDS_PATH.name}...")
    with open(IDS_PATH, "w") as f:
        json.dump(chunk_ids, f)

    print(f"\nDone. {len(chunk_ids)} entry points indexed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
