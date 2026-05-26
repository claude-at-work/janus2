# JANUS — Just Another Natural Ultimate System

## What It Is

Terminal assistant where a conversational LLM signals shell actions for human approval. Underneath: a symbolic link graph where meaning travels through typed relationships between chunks, not vector similarity.

## The Shape

```
User speaks naturally
  → LLM interprets, surfaces <ACTION> blocks
  → User approves/rejects/edits
  → Executor runs in sandbox
  → Result injected back into conversation
  → Memory compresses, exports, recalls across sessions
  → Symlink graph learns the geometry of the conversation archive
```

## History

The project traced this path:

1. **Natural language terminal wrapper** — sandboxed directory, conversational I/O, Ollama inference
2. **Personality system** — TaskRabbit (minimal), Jeffrey (butler), Co-Captain (partner with opinions, default). Co-Captain values honesty over agreeability.
3. **3-layer memory** — L1 in-memory, L2 rolling JSON journal (`~/.janus/context.json`), L3 bud FAISS vector index. Compression at 40 turns. Session export in Claude conversation format for bud indexing.
4. **Bud integration** — forked bud into the project for customization. Semantic recall via embeddings. No Voyage AI — direct text embeddings via Ollama.
5. **Symlink index** — graph of symbolic links between chunks. Meaning through traversal paths, not cosine similarity. Schema dimensions: geometry, coherence, texture, terrain.
6. **Hyper-symbolic MLP** — multi-depth link generation. Level 1: base links (analogy, contrast, extension). Level 2: meta-links. Level 3: abstract patterns. Lossy links at higher depths.
7. **Discover → chunk pipeline** — `bud/stages/discover.py` builds concept map from archive geometry. `bud/stages/chunk.py` chunks with LLM, emits `link_to_next` edges. Tags come from chunking, not from a separate tagger.
8. **Slicer/Splicer** — SlicingMLP predicts which chunks to extract for a query. BlobSplicingMLP/SplicingMLP predicts how to recombine them.
9. **MLP trainer** — backprop with SGD+momentum, weight persistence to JSON, early stopping.
10. **Aesthetic loss** — phi-resonance across depth confidences. The golden ratio as loss function component. Fractal self-similarity measurement. The network spirals toward phi without arriving because phi is irrational. This is not post-hoc scoring — it's part of what "good link" means during training.
11. **Tagger removed from training pipeline** — the MLP learns directly from embeddings and chunker-provided tags. The lexical heuristic tagger is dead weight.
12. **Positional edges from chunker** — `link_to_next` from `bud/stages/chunk.py` feeds directly into training data as ground truth. The MLP only needs to discover non-adjacent, non-obvious relationships.

## Current State

The pipeline is: `discover → chunk → train → symlink index`. The aesthetic loss is wired. The tagger dependency is removed. The positional edges flow from chunker to trainer.

**Open seams:**
- Orchestration wrapper around slicer/splicer (when to slice vs splice)
- Bridge from trained weights back to live recall (recall path still goes through raw FAISS, not through the symbolic link graph)
- `bridges.py`, `vector_compress.py`, `tagger.py` — dead modules to remove
- Two VectorStore / EmbeddingClient implementations to consolidate

## Quick Start

```bash
bash setup.sh        # wizard: model, personality, sandbox
python3 -m janus     # REPL
```

Requires: Python 3.10+, Ollama running locally.

## Modules

```
shell.py             REPL, action approval, ! commands
llm.py              Ollama: think(), agent_fix(), summarize()
executor.py          Sandboxed subprocess, safety blocks
memory.py            3-layer memory, export, compression, recall
personalities.py     3 presets + custom
symlink_index.py     Symbolic link graph, HyperSymbolicMLP, traversal
trainer.py           TrainableMLP, aesthetic loss (phi-resonance), weight persistence
vector.py            FAISS VectorStore, EmbeddingClient, SessionIndexer
scratch.py           Scratch pad with criticality levels
middleware.py        JanusBridge, JanusSessionManager
config.py            Env vars

bud/stages/
  discover.py        Iterative pattern discovery, concept map
  chunk.py           LLM-based chunking with link_to_next edges
  embed.py           Embedding stage
  index.py           FAISS index building
```

## Config

```
OLLAMA_HOST=http://localhost:11434
JANUS_MODEL=qwen3-vl:235b-cloud
JANUS_SANDBOX=/root/janus/sandbox
```

## Dependencies

Runtime: `faiss-cpu numpy click rich pyyaml requests mcp prompt-toolkit anthropic`
External: Ollama with a pulled model
