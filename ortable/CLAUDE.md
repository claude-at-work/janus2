# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A self-contained, portable training package for the JANUS Link MLP. **No JANUS imports** — ships to any machine with numpy. The only file that matters for development is `train_remote.py`.

See `INSTRUCTIONS.md` for the user-facing quick-start guide.

## Running Training

```bash
# GPU (fastest — ~2min on T4 for full corpus)
pip install numpy sentence-transformers torch
python train_remote.py --backend sentence_transformers

# Voyage AI API (~$0.02, no GPU)
pip install numpy
export VOYAGE_API_KEY="your-key"
python train_remote.py --backend voyage

# Ollama (slow, ~5 hours)
python train_remote.py --backend ollama
```

Key flags: `--epochs N`, `--max-turns N`, `--max-pairs N`, `--fresh` (resets `data/embed_checkpoint.json`, keeps `data/embedding_cache.jsonl`).

Training is safely resumable — the embedding cache is append-only and never re-embeds already-seen text. If interrupted, re-run the same command; embedding phase is instant for cached turns.

## Deploying to Remote GPU

```bash
bash telepad.sh   # interactive wizard: prompts for IP/user/port, handles SSH key, uploads, installs deps
```

After training, pull weights back:
```bash
scp -i ~/.ssh/janus_telepad user@ip:~/portable/link_mlp_trained.json ../link_mlp_trained.json
# also pull cache to avoid re-embedding next run:
scp -i ~/.ssh/janus_telepad user@ip:~/portable/data/embedding_cache.jsonl data/embedding_cache.jsonl
```

## Architecture

**Pipeline**: corpus → embed (384-dim → stride-sampled to 64-dim) → pair generation → MLP training → JSON weights

**NumpyMLP**: `128 → [64, 32] → 8` (7 link types + confidence scalar). No PyTorch dependency — pure numpy with SGD+momentum + cosine LR decay. Early stopping at patience=10 epochs without improvement.

**Link types**: `analogy`, `contrast`, `extension`, `context`, `resolution`, `metaphor`, `symbol`. Labels are assigned from observable signals (speaker transition × cosine similarity for positional pairs; cross-conversation boundary × cosine thresholds for discovered pairs) — not hand-labeled.

**Pair generation** (two sources):
- *Positional*: adjacent turns within same conversation; label derived from speaker transition × cosine similarity
- *Discovered*: random non-adjacent pairs; label from cross-conversation/same-conversation cosine thresholds

**Pair compression**: epochs ≥ 5, pairs the MLP predicts correctly with high confidence get pruned progressively (keeping `keep_ratio^(epoch/10)` of collapsed pairs). Prevents wasted compute on "solved" examples.

**Phi-resonance loss**: cross-entropy + confidence MSE + aesthetic term that pulls confidence ratios across depth levels toward φ (1.618...). Because φ is irrational, the loss never fully zeros — it spirals asymptotically.

## Output

- `link_mlp_trained.json` — trained weights (copy to `janus/link_mlp_trained.json`)
- `link_mlp_trained.backup.json` — same, written first as safety copy
- `data/embedding_cache.jsonl` — hash-keyed cache; bring this home too to avoid re-embedding
- `data/embed_checkpoint.json` — embedding frontier tracker (reset with `--fresh`, not the cache)

Weight format is compatible with `trainer.py:load_trained_weights_into_mlp()` and `symlink_index.py:SimpleMLP` in the parent repo.

## Diagnostic Tool

`data/embed_and_predict.py` — quick script to test a trained model: embeds up to 40 turns from the corpus via Ollama, pairs them, and runs them through `link_mlp_trained.json` to show predicted link types and confidence scores. Requires Ollama running locally with `all-minilm:33m` pulled. No flags — edit `MAX_TURNS` at the top if needed.

```bash
python data/embed_and_predict.py
```

## Data Format

`data/conversations_*.jsonl` — one JSON object per line, each with a `turns` array. Each turn has `sender`, `text`, `conv` (conversation name). Turns ≤ 20 chars are skipped. Text is truncated to 512 chars before embedding and hashing. Cache keys are the first 16 hex chars of the SHA-256 of the truncated text.

The corpus contains 6349 turns total. The embedding cache ships with 248 pre-computed vectors from a prior run.
