# JANUS Link MLP — Portable Training

## What's in the box

```
portable/
├── train_remote.py              # self-contained training script (no JANUS imports)
├── INSTRUCTIONS.md              # you are here
├── data/
│   ├── conversations_*.jsonl    # the corpus (6349 turns)
│   └── embedding_cache.jsonl    # 248 pre-computed vectors from prior run
├── link_mlp_trained.backup.json # 250-turn fragment weights (baseline)
└── link_mlp_trained.json        # ← output goes here
```

## Quick start

### Option A: GPU box with sentence-transformers (fastest)

```bash
pip install numpy sentence-transformers torch
python train_remote.py --backend sentence_transformers
```

This uses `all-MiniLM-L6-v2` on GPU. Same model family as the local Ollama
`all-minilm:33m`. Embeds 6349 turns in ~2 minutes on a T4, seconds on an A100.

### Option B: Voyage AI API (no GPU needed)

```bash
pip install numpy
export VOYAGE_API_KEY="your-key-here"
python train_remote.py --backend voyage
```

Uses `voyage-3-lite`. Batch API, fast. Costs ~$0.02 for the full corpus.

### Option C: Ollama (local, slow)

```bash
pip install numpy
# assumes ollama running at localhost:11434 with all-minilm:33m pulled
python train_remote.py --backend ollama
```

~5 hours for the full corpus. Not recommended for the full run.

## What it does

1. Walks all `conversations_*.jsonl` files, extracts turns > 20 chars
2. Embeds each turn (384-dim → stride-sampled to 64-dim)
3. Caches embeddings to `embedding_cache.jsonl` (never re-embeds)
4. Labels adjacent pairs from speaker transitions × cosine similarity
5. Labels non-adjacent pairs from cosine distance × conversation boundaries
6. Trains a 128→[64,32]→8 MLP with SGD+momentum, phi-resonance aesthetic loss
7. Pair compression prunes collapsed pairs as training progresses
8. Saves weights to `link_mlp_trained.json` + backup

## Flags

```
--epochs N         Training epochs (default: 50)
--max-turns N      Cap on corpus turns (default: 6349, the full corpus)
--max-pairs N      Cap on training pairs (default: 10000)
--fresh            Reset checkpoint (cache still used)
--backend X        sentence_transformers | voyage | ollama
```

## Resuming

If interrupted, just re-run the same command. The embedding cache means
already-embedded turns load instantly. Training restarts from scratch
(weights are small, training is fast — the expensive part is embedding).

## Bringing weights home

Copy `link_mlp_trained.json` back to the JANUS repo at:

```
janus/link_mlp_trained.json
```

The format is compatible with `trainer.py:load_trained_weights_into_mlp()`
and `symlink_index.py:SimpleMLP`.
