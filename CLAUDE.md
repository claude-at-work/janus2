# CLAUDE.md — JANUS 2

This is a fork and continuation of JANUS, begun May 2026.

The original lives at `~/janus/`. This fork was started not because the original was broken, but because it had been shelved and moved enough times that the live data (`~/.janus/`) was gone, and the moment of dusting it off felt like the right moment to do it with intention — to bring everything over deliberately, understand what each piece is and why it exists, and build forward from a clean foundation.

This CLAUDE.md is not a specification. It's a living document. Update it when it goes stale.

---

## What This Is

JANUS is a terminal assistant with a two-component REPL (conversational LLM + terminal agent) sitting on top of a **symbolic graph memory**. The central commitment: **meaning lives in edges, not nodes**. Retrieval is graph traversal through typed relationships — analogy, contrast, extension, context, resolution, metaphor, symbol — not nearest-neighbor vector lookup.

The graph is trained on conversation history. The MLP learns the shape of relationships between chunks. The traversal IS the memory.

---

## Founding Hypotheses for JANUS 2

These are the directions that feel alive as of this fork. They are hypotheses, not tasks.

**1. Thinking blocks as first-class nodes**

The old corpus (`ortable/data/old_corpus/`) contains 7,112 turns from Claude conversation exports, including `thinking` and `dropped_blocks` fields — extended reasoning traces. These have never been used. The hypothesis: thinking blocks are a different epistemic register than surface text. Surface text is presentational. Thinking is exploratory — attention in motion before it settles. A graph that includes both registers would produce structurally richer traversals, and the link types JANUS infers from the outside are often *explicit* inside thinking traces.

**2. Better label assignment**

The current MLP training labels are derived from cosine similarity × speaker transition — a heuristic proxy. The result: 51% symbol, 31% metaphor, <1% extension. This feels wrong for a corpus of actual thinking. Thinking blocks contain cognitive operation markers that could serve as ground-truth labels: causal language → extension, comparative language → analogy/contrast, hedging + resolution → resolution.

**3. Dream mode**

The narrate state machinery (stack, residue, trajectory signature) is in place. The next step named in the journal: the graph querying itself without a user prompt — starting from residue, walking freely, updating the stack. The hypothesis: residue would drift toward the graph's structural center of mass, the most *connected* territory rather than the most frequently visited.

**4. Corpus-derived connective tissue**

The current `connective_tissue.json` is 980 bytes — a handful of generic phrases per link type. The tissue could instead be derived from the corpus itself: actual transitional moments from Tyler's thinking, clustered by link type. Then the voiced traversal sounds like the corpus talking about itself.

---

## Architecture (inherited from JANUS 1)

### Memory: 3 Layers

- **L1** — In-memory chat history (live session)
- **L2** — `~/.janus/context.json` (rolling 50 turns, injects 20 on startup)
- **L3** — Symlink graph (typed-edge traversal) + FAISS fallback

### The Pipeline

```
Conversations (JSONL)
    ↓
train_remote.py → embed → generate pairs → train Link MLP (phi-resonance loss)
    ↓
link_mlp_trained.json
    ↓
build_graph.py → symlink_index.json (chunks + typed edges)
    ↓
build_entry_index.py → entry_embeddings.npy + entry_chunk_ids.json
    ↓
~/.janus/ (live system)
```

### Narrate State (most recent layer)

Three-level walking memory:
- `session_context` — running vector accumulated during current walk
- `stack` — up to 8 `ContextImpression` objects across walks/sessions (each has a vector + `TrajectorySignature`)
- `residue` — crystallized long-term vector, updated when impressions consolidate

Consolidation criteria (2 of 3): cosine similarity to residue ≥ 0.6, touch count ≥ 2, rare link types in trajectory.

### The Conductor

`conduct_recall()` routes queries to:
- `splice_recall` — relational intent (shows graph topology)
- `recall_as_context` — extractive intent (raw chunks)

### Phi

φ = 1.6180339887... governs: traversal spinner timing, aesthetic loss, context vector blending, depth weight fade, momentum tracking. The loss that includes φ never zeros because φ is irrational. This is a design commitment, not a bug.

---

## Link Types

`analogy`, `contrast`, `extension`, `context`, `resolution`, `metaphor`, `symbol`

## Schema Dimensions (chunk tags)

| Dimension | Values |
|-----------|--------|
| geometry | linear, recursive, spiral, bifurcating |
| coherence | tight, loose, fragmented, emergent |
| texture | dense, sparse, lyrical, technical |
| terrain | conceptual, emotional, procedural, speculative |

---

## What NOT to Touch (inherited from JANUS 1)

- The aesthetic loss is tuned. Don't add complexity to it.
- Phi is not decorative. It governs everything irrational about this system.
- The narrate walk is the most distinctive feature. Preserve it.
- Don't resurrect the tagger. The MLP replaced it.

---

## Data Layout

```
~/.janus2/          (runtime data, not in repo)
├── index/          symlink graph, entry embeddings
├── weights/        link_mlp, connector_mlp, flow_mlp, connective_tissue
├── data/           conversations, embedding cache
├── logs/           narrate_history.jsonl
├── sessions/       session exports
├── context.json    rolling journal
└── scratch.json    scratch pad

ortable/            portable training artifact (in repo)
    data/           corpus + caches + built artifacts
    train_remote.py
    build_graph.py
    build_entry_index.py
```

---

## Documentation That Must Travel With This Project

The history of this project is load-bearing. These documents are not peripheral:

- `docs/collaborative-journal.md` — the working relationship, verbatim
- `janus/VECTOR.md` — where the system was pointed and what not to touch
- `janus/README.md` — the full development history, step by step
- `docs/superpowers/` — implementation plans and specs for every major feature
- `issues.md` — known problems and open questions

If those get lost, the code becomes mechanisms without philosophy.

---

## Navigation

| What | Where |
|------|-------|
| Collaborative journal | `docs/collaborative-journal.md` |
| Original direction vector | `janus/VECTOR.md` |
| Full history | `janus/README.md` |
| Voice output plan | `docs/superpowers/plans/2026-04-05-voice-output.md` |
| Narrate context plan | `docs/superpowers/plans/2026-04-07-narrate-context-awareness.md` |
| Open issues | `issues.md` |
