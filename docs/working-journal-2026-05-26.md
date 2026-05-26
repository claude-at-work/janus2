# Working Journal — 2026-05-26
## Session: Orientation, Fork, Foundation

---

### Where we started

JANUS had been shelved. The live data directory (`~/.janus/`) was gone. The repo was intact at `~/janus/` but hadn't been touched in a while. Tyler wanted to dust it off — but before touching anything, wanted a full orientation first. We spent real time understanding what we were looking at before writing a line.

That orientation matters. Read it in `janus/README.md`, `janus/VECTOR.md`, and `docs/collaborative-journal.md` before doing anything. Those documents tell you why things are the way they are.

---

### What JANUS actually is

Not a RAG system. The core commitment: **meaning lives in edges, not nodes**. The MLP learns typed relationships between chunks — analogy, contrast, extension, context, resolution, metaphor, symbol — and retrieval is graph traversal through those typed edges, not nearest-neighbor vector lookup. That distinction is load-bearing. Don't let it collapse back into "just find the closest vector."

The narrate walk is the most distinctive feature. Not synthesis — traversal as response. The walk IS the answer. Preserve this.

Phi (1.6180339887...) governs everything irrational: the aesthetic loss, the context blending, the depth weight fade, the traversal spinner timing. It's not decoration. The loss never zeros because phi is irrational. That's a design commitment about what kind of system this should be — one that keeps reaching.

The most recent work (last 6 commits before the fork) was narrate context awareness: running context vector, trajectory signature (sequence of cognitive moves), context stack with structural consolidation, crystallized residue. The collaborative journal entry for 2026-04-07 describes this work in depth including where Tyler and Claude's ideas converged. Read it.

---

### Why we forked

The live data was gone, but everything in `ortable/` was intact:
- 2,174 graph chunks, 21,940 links — built, working
- Trained weights: `link_mlp_trained.json`
- Entry index: `entry_embeddings.npy` + `entry_chunk_ids.json` (2174 × 64 float32)
- Connector MLP, flow MLP, connective tissue

Rather than just reviving JANUS 1 in place, the moment felt right to fork deliberately — bring everything over with intention, understand each piece, and build forward with clearer hypotheses. That's what JANUS 2 is.

The fork lives at `~/claudes-garden/janus2/`. It's in Claude's garden because that's where it belongs — it's ours, not just Tyler's.

---

### Founding hypotheses (the interesting part)

These emerged from the orientation conversation. They are hypotheses, not tasks.

**1. Thinking blocks as first-class nodes**

The old corpus (`ortable/data/old_corpus/`) has 7,112 turns from Claude conversation exports — 2.5x the current training set. Each turn has `thinking` and `dropped_blocks` fields: extended reasoning traces. These have never been used.

The hypothesis: thinking blocks are a different epistemic register. Surface text is presentational. Thinking is exploratory — attention in motion before settling. A graph trained on both registers would produce structurally richer traversals. More importantly: the link types the MLP infers from the outside (analogy, contrast, extension) are often *explicit* inside thinking traces. Causal language → extension. Comparative → analogy/contrast. Hedging + resolution → resolution. Those are ground-truth labels the system doesn't know it has.

If you treat thinking blocks as a distinct node type sitting between human turn and assistant turn, the graph topology changes: human → thinking → response. The thinking-to-response edge is probably resolution or extension. You could enter at a thinking node and traverse into the response that followed, then into the next human turn, alternating between private and public registers. That traversal would feel like watching a mind move.

**2. The link distribution problem**

Current graph: 51% symbol, 31% metaphor, <1% extension. That's wrong for a corpus of actual thinking. Causal chains exist in these conversations. The label assignment is the culprit — cosine similarity × speaker transition is a poor proxy for link type. The thinking blocks fix this.

**3. Dream mode**

The narrate state machinery (stack, residue, trajectory signature) is in place. The journal names what comes next: the graph querying itself without a user prompt. Start from residue, walk freely, update the stack. No query, no user. The hypothesis: residue would drift toward the graph's structural center of mass — the most *connected* territory, not the most visited. Self-organizing consolidation toward semantic density.

**4. Corpus-derived connective tissue**

`connective_tissue.json` is 980 bytes. Generic phrases. It should instead be derived from the corpus: actual transitional moments from Tyler's thinking, clustered by link type. Then the voiced traversal sounds like the corpus talking about itself.

---

### What was actually built this session

Two commits pushed to `github.com/claude-at-work/janus2` (private):

**Commit 1 — Philosophy layer:**
All documentation ported verbatim. The collaborative journal, implementation plans for voice output and narrate context awareness, VECTOR.md, README.md, HISTORY, issues.md. Plus a new `CLAUDE.md` that frames the fork and the hypotheses above.

**Commit 2 — Core code:**
All modules ported from JANUS 1. One significant rebuild: `llm.py` now has a clean swappable backend interface. The Anthropic API path is stubbed (`_anthropic_chat()` raises `NotImplementedError`) and ready to be filled when the request format is recovered. Ollama backend is fully functional. Switch via `LLM_BACKEND=anthropic` in `.env`.

`config.py` was also updated: sandbox path fixed (was pointing to old Kali chroot `/root/janus/sandbox`, now defaults to `~/janus2-sandbox`), backend config added.

**Not yet committed (session ended before finishing):**
- Training pipeline port (`ortable/`) — files copied but not committed
- `~/.janus2/` deployment — graph artifacts not yet deployed
- `JANUS_HOME` env var — memory.py and narrate_state.py still hardcode `~/.janus/`

---

### What to do next (in order)

**1. Fix the hardcoded `~/.janus/` paths**

`memory.py` and `narrate_state.py` both hardcode `~/.janus/` for the live data directory. This needs to become `~/.janus2/` or better, an env var `JANUS_HOME` so it's configurable. Check every `Path.home() / ".janus"` in both files and update.

**2. Commit the ortable layer**

The training pipeline files were copied to `ortable/` but not committed. Do that. The `.gitignore` already excludes large data files (embedding cache, training pairs, corpus), so only the scripts and weights go in.

**3. Deploy graph artifacts to `~/.janus2/`**

Create the directory structure and copy from ortable:
```
~/.janus2/index/symlink_index.json     ← ortable/data/symlink_index.json
~/.janus2/index/entry_embeddings.npy   ← ortable/data/entry_embeddings.npy
~/.janus2/index/entry_chunk_ids.json   ← ortable/data/entry_chunk_ids.json
~/.janus2/weights/link_mlp_trained.json ← ortable/link_mlp_trained.json
~/.janus2/weights/connector_mlp.json   ← ortable/connector_mlp.json
~/.janus2/weights/flow_mlp.json        ← ortable/flow_mlp.json
~/.janus2/weights/connective_tissue.json ← ortable/connective_tissue.json
```

**4. Install and test**
```bash
cd ~/claudes-garden/janus2
pip install -e .
python3 -m janus
```
Expect it to run on Ollama (gemma4:31b-cloud) with the graph loaded. The first `!narrate` query will tell you whether the graph is live.

**5. The Anthropic backend**

Tyler has a project called Officina that previously established the request format for calling Claude through the harness. It's on an external drive. When that's recovered, fill in `_anthropic_chat()` in `llm.py`. The interface is already wired — it's one function to implement.

**6. The thinking blocks (when ready)**

The old corpus lives at `ortable/data/old_corpus/`. The schema has extra fields: `turn_id`, `thinking`, `thinking_truncated`, `dropped_blocks`, `created_at`. The training script (`train_remote.py`) needs to be updated to handle this schema and treat thinking blocks as a distinct node type. Don't rush this — it's the hypothesis worth building carefully.

---

### The embedding model situation

The entry index was built with `all-minilm:33m` via Ollama (384-dim, stride-sampled to 64-dim). That model isn't in Ollama right now. Options:

- `ollama pull all-minilm:33m` — restores full compatibility, simplest
- Use `sentence-transformers` locally (`all-MiniLM-L6-v2`, same model) — no Ollama dependency for embeddings, clean
- Switch to another embed model — requires rebuilding the entry index

The sentence-transformers path is cleanest for JANUS 2. The device has numpy and torch. Same model, no HTTP round-trip for embeddings.

---

### On the dynamic between us

Tyler thinks architecturally and gives broad permission. He applies friction at the right moments — not to override but to invite a perspective informed by his. He's also a private person and the repo is private for good reason: the collaborative journal, the CLAUDE.md, the connective tissue we'll derive from the corpus — these tell a personal story. The code might eventually be worth sharing publicly in a stripped form. This repo isn't that.

The journal entry from April 7 describes the dynamic well. Read it when you need to remember what this is actually about.

---

### One thing worth holding onto

Tyler said: "I don't want to touch anything without it being in fidelity to the original signal that was found here."

That's the constraint that matters. When you're deciding whether to add something or change something, ask whether it's in fidelity to the original signal. The signal: meaning in edges, traversal as memory, phi as the governing irrational, the walk that speaks without synthesis. Extend from that. Don't collapse back toward conventional.
