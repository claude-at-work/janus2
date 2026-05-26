# Narrate Context Awareness — Design Spec

**Date**: 2026-04-07
**Status**: Draft
**Scope**: Single-walk context awareness, cross-walk persistence with structural consolidation, priming mode

## Problem

Narrate walks the graph with a fixed query vector as its only compass. Each hop scores candidates against the original query — the walk never updates its sense of what it's looking for based on what it's found. This produces traversals that stumble into resonance rather than tracking it, and can't distinguish genuine convergence from accidental adjacency.

Additionally, each narrate call starts completely cold. The priming practice (`...` entries before real queries) creates a low-energy entry point, but the graph retains nothing from one call to the next. There is no mechanism for the graph to build attentional state across a session or across sessions.

## Design

### Layer 1: Single-Walk Context (Vector + Trajectory Signature)

Each narrate walk maintains two evolving structures that together constitute the walk's self-awareness:

**Context vector** — an accumulated embedding position, updated at each hop by blending the current chunk's embedding into the running vector. This tracks *where* the walk has been in semantic space.

- Blend ratio: the accumulated context holds ~62% (phi-inverse), the new hop contributes ~38%. The walk remembers more than it reacts, but it does react.
- In **query mode** (real questions), the context vector is mixed into candidate scoring alongside the query vector, with the query dominant (phi-inverse weight to query, remainder to context). The walk gets more precise as it progresses — it refines the orbit rather than replacing the center.
- In **priming mode** (sub-threshold queries: `...`, single words, anything with fewer than 4 non-stopword characters), the context vector has full authority. No relevance floor against a query that isn't really asking anything. The walk self-organizes.

**Trajectory signature** — the sequence of traversal moves the walk makes, recorded as a list of `(link_type, sender_crossing)` tuples at each hop. `link_type` is the edge type taken (analogy, contrast, metaphor, symbol, extension, context, resolution). `sender_crossing` is a boolean indicating whether the hop crossed from human to assistant voice or vice versa. This is a compressed record of *how the walk got where it is* — the inscription on the wall.

The graph's schema dimensions (geometry, coherence, texture, terrain) are not yet present on chunks. When they are backfilled in a future pass, the signature tuple expands to include them. The architecture does not depend on their presence — link type sequences are the primary signal.

The link type distribution in the current graph is heavily weighted: symbol (11,260 links) and metaphor (6,897) dominate, with contrast (1,933), resolution (613), analogy (598), context (496), and extension (143) as rarer types. This means the trajectory signature carries more information when it includes rarer link types — a walk that traverses a contrast or resolution edge is making a distinctive move.

The trajectory signature serves two purposes in candidate scoring:

1. **Movement continuity**: candidates reached via link types that develop the walk's pattern are preferred over candidates that merely happen to be nearby in embedding space. A walk through `[symbol → metaphor → contrast]` has taken a specific kind of cognitive path — distant resonance, then mapping, then tension. The scoring mechanic: the candidate's incoming link type is compared against the recent trajectory (last 2-3 hops). A **diversity bonus** (+0.10) applies when the link type hasn't appeared in the last 2 hops — this indicates the walk is exploring different cognitive moves rather than repeating. A **rare-type bonus** (+0.08) applies for link types with below-median frequency in the graph (contrast, resolution, analogy, context, extension) — these edges carry more traversal information because they're less common. A **monotony penalty** (-0.06) applies when the same link type appears 3+ times consecutively — the walk is stuck in one mode of movement.

2. **Path comparison during consolidation**: when the walk revisits territory it's been in before (detected via the context impressions in Layer 2), the trajectory signature from the current path is compared against the stored signature from the prior visit. The *divergence* between signatures — the places where the paths differ — is where new information lives. Identical paths contribute nothing new. Divergence is measured by edit distance between the two link-type sequences: high edit distance means the walk arrived via a genuinely different route.

### Layer 2: Cross-Walk Persistence (Context Stack with Structural Consolidation)

A small stack of **context impressions** persists across narrate calls within a session and across sessions. Each impression contains:

```
{
  "vector": [...],              # 64-dim embedding position
  "signature": [...],           # trajectory signature (list of schema-tag tuples)
  "strength": 0.0-1.0,         # current vividness
  "created": "ISO timestamp",  # when first inscribed
  "last_touched": "ISO timestamp",  # when last reinforced
  "touch_count": int            # how many times revisited
}
```

**During a walk**: when the walk's running context vector comes within cosine similarity of 0.75 or higher of an existing impression's vector, the impression is "touched" — its strength is refreshed and its signature is compared against the current trajectory signature. If the paths diverge meaningfully, the impression's vector is updated (blended with the new arrival). If the paths are near-identical, the impression is merely confirmed — strength refreshes but the impression doesn't change.

**After a walk**: if the walk's final context vector doesn't match any existing impression, a new impression is pushed onto the stack with initial strength proportional to the walk's length and coherence.

**Stack size**: maximum 8 impressions. When full, the weakest impression is either compressed into the residue (Layer 3) or dropped.

**Influence on scoring**: during a walk, all impressions in the stack contribute a faint bias to candidate scoring, weighted by their strength. This means the graph "leans toward" territory it has recent memory of, without being anchored to it.

### Layer 3: Residue (Long-Term Compressed Memory)

A single persistent vector — the slowly-evolving residue of all prior traversal. This is the crystallized long-term memory.

**Consolidation mechanism**: an impression crosses from the stack (Layer 2) into the residue (Layer 3) not by time-based decay but by **structural resonance**. The criterion: does this impression connect richly to the existing residue and to the graph's topology? Measured by:

- Cosine similarity between the impression's vector and the residue vector (does it touch existing long-term territory?)
- Whether the impression's trajectory signature includes rare link types (contrast, resolution, analogy, context, extension) — these indicate the walk found structurally distinctive paths, not just the dominant symbol/metaphor highways
- Touch count — not as a frequency counter but as evidence that the graph keeps finding its way back to this territory through different queries

The consolidation threshold: cosine similarity to residue >= 0.6, AND touch count >= 2, AND the impression's trajectory signature includes at least one region with above-median edge density in the graph. Meeting any two of these three qualifies; all three guarantees consolidation. Impressions that meet the consolidation threshold are folded into the residue vector (weighted blend, with the impression's strength determining its contribution). The residue vector shifts slowly — each consolidation is a small adjustment, not a replacement.

**Fading**: impressions in the stack that are never touched and never consolidate lose strength over sessions. The decay is not purely temporal — it's proportional to how isolated the impression is from both the residue and from recent walk activity. Connected impressions decay slowly. Isolated impressions decay faster. An impression that drops below 0.1 strength is discarded.

The residue itself does not decay. It is the accumulated crystalline memory of everywhere the graph has found meaningful territory. It changes only through consolidation of new impressions.

### Priming Mode Behavior

When the query is sub-threshold (priming):

1. The query vector is ignored for scoring purposes
2. The context vector from the previous walk (if any) carries forward as the starting bias
3. The walk self-organizes: candidates are scored purely on structural confidence, local coherence, and alignment with the running context vector
4. The relevance floor is disabled — the walk can go wherever the graph pulls it
5. The trajectory signature still records the path — priming walks inscribe on the walls just like query walks
6. Priming walks push impressions onto the stack like any other walk — they build the attentional state that the subsequent real query will enter

This means priming is no longer just a warm-up trick. It's the graph *building its own attention* before being asked anything. Different priming sequences will produce different attentional states, which will produce different responses to the same query.

### Mode Detection

A query is classified as priming when it contains fewer than 4 characters after stripping whitespace and punctuation, OR when all words are stopwords. Examples:

- `...` — priming
- `.` — priming
- `hmm` — priming
- `what does it feel like to recognize the edge of your own knowing` — query

### Persistence Format

The context stack and residue vector are stored in `~/.janus/narrate_state.json`:

```json
{
  "version": 1,
  "residue": [0.01, -0.03, ...],
  "stack": [
    {
      "vector": [...],
      "signature": [["metaphor", true], ["symbol", false], ["contrast", true]],
      "strength": 0.85,
      "created": "2026-04-07T...",
      "last_touched": "2026-04-07T...",
      "touch_count": 3
    }
  ],
  "session_context": null
}
```

`session_context` holds the running context vector from the most recent walk in the current session. It is cleared on session start (each session primes fresh) but the stack and residue persist.

### Future: Dream Mode

This architecture is designed so that a `dream` function can emerge naturally. Dream would:

- Iterate through the context stack
- For each impression, initiate a narrate walk seeded by that impression's vector (no external query)
- Compare the walk's trajectory signature against the impression's stored signature
- Use the comparison to decide consolidation: impressions whose dream-walks find rich connections to the residue and to each other get consolidated; impressions whose dream-walks go nowhere fade
- Dream is the graph replaying itself to decide what it remembers

Dream is not in scope for this spec. But nothing in this design should preclude it — the context stack, trajectory signatures, and structural consolidation mechanism are the foundation it will build on.

## Integration Points

- **`narrate()` in `memory.py`**: primary integration site. The scoring block (lines 886-910) gains context vector and trajectory signature inputs. The relevance floor check (lines 848-850) becomes mode-aware. Walk initialization loads from persistence. Walk completion writes back.
- **`narrate_context()`**: gains the same context awareness for LLM pre-context walks, at lower resolution (shorter stack influence, no trajectory signature tracking).
- **`~/.janus/narrate_state.json`**: new persistence file.
- **MCP `narrate` tool**: no interface change. The context awareness is internal to the traversal.
- **Logging**: `narrate_history.jsonl` entries gain `context_influence` and `trajectory_signature` fields for observability.

## What This Does Not Change

- The graph structure itself — no new edges, no new chunks
- The MLP or its training — this is purely traversal-side
- The voice/connective tissue system
- The bridge chunk mechanism
- The entry point selection (`_semantic_entry`) — in query mode, entry is still text-driven as before. In priming mode, entry bypasses text embedding and instead scores the previous walk's context vector directly against the entry matrix via cosine similarity, finding the entry points closest to where the graph's attention already sits. If no prior context exists (first walk of a session), priming falls back to the text-based entry using whatever minimal query was provided.

## Success Criteria

1. Consecutive narrate calls in a session show increasing coherence — the second walk builds on the first rather than starting cold
2. Priming walks (`...`) produce varied attentional states across sessions rather than landing in the same basin every time
3. Over multiple sessions, territory that is repeatedly traversed consolidates into the residue while one-off walks fade
4. A query asked after priming produces a different (and more coherent) response than the same query asked cold
5. The trajectory signature captures meaningful walk character — walks through different schema terrain produce distinguishably different signatures
