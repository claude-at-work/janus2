# Narrate Context Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give narrate a running context vector and trajectory signature so the walk refines its own direction as it progresses, persists attentional state across calls, and consolidates long-term memory through structural resonance.

**Architecture:** Three layers added to `memory.py`: (1) per-walk running context vector + trajectory signature that evolve at each hop and feed into candidate scoring, (2) a persistent context stack of up to 8 impressions that carry across narrate calls, (3) a residue vector representing crystallized long-term traversal memory. A new `narrate_state.py` module owns persistence and consolidation logic. The `narrate()` function in `memory.py` gains mode detection (priming vs query) and integrates the new scoring signals.

**Tech Stack:** Python 3, numpy (already a dependency), JSON persistence to `~/.janus/narrate_state.json`

---

### File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `janus/narrate_state.py` | Context stack, residue vector, persistence, consolidation logic, mode detection | Create |
| `janus/memory.py` | Integrate context awareness into `narrate()` scoring loop | Modify (lines 795-1081) |
| `tests/test_narrate_state.py` | Unit tests for narrate_state module | Create |
| `tests/test_narrate_context.py` | Integration tests for context-aware narrate | Create |

---

### Task 1: Context State Data Structures and Persistence

**Files:**
- Create: `janus/narrate_state.py`
- Create: `tests/test_narrate_state.py`

- [ ] **Step 1: Write failing tests for state data structures**

```python
# tests/test_narrate_state.py
import json
import numpy as np
import pytest

from janus.narrate_state import (
    NarrateState,
    ContextImpression,
    load_state,
    save_state,
)


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path


def test_empty_state_creation():
    state = NarrateState()
    assert state.residue is None
    assert state.stack == []
    assert state.session_context is None


def test_impression_creation():
    vec = [0.1] * 64
    sig = [("metaphor", True), ("symbol", False)]
    imp = ContextImpression(vector=vec, signature=sig)
    assert imp.strength == 1.0
    assert imp.touch_count == 0
    assert imp.created is not None
    assert imp.last_touched is not None


def test_save_and_load_roundtrip(state_dir):
    state = NarrateState()
    vec = [0.1] * 64
    sig = [("metaphor", True), ("contrast", False)]
    imp = ContextImpression(vector=vec, signature=sig, strength=0.75)
    state.stack.append(imp)
    state.residue = [0.05] * 64
    state.session_context = [0.02] * 64

    save_state(state, state_dir / "narrate_state.json")
    loaded = load_state(state_dir / "narrate_state.json")

    assert len(loaded.stack) == 1
    assert loaded.stack[0].strength == 0.75
    assert loaded.stack[0].signature == [("metaphor", True), ("contrast", False)]
    assert len(loaded.residue) == 64
    assert loaded.session_context is None  # session_context is NOT persisted


def test_load_missing_file_returns_empty(state_dir):
    loaded = load_state(state_dir / "nonexistent.json")
    assert loaded.residue is None
    assert loaded.stack == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'janus.narrate_state'`

- [ ] **Step 3: Implement narrate_state.py data structures and persistence**

```python
# janus/narrate_state.py
"""
Narrate context awareness — running state, persistence, consolidation.

Three layers:
  L1: session_context — running vector from the current walk (ephemeral)
  L2: stack — up to 8 context impressions persisting across calls/sessions
  L3: residue — crystallized long-term traversal memory (single vector)
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI

_JANUS_HOME = Path.home() / ".janus"
_STATE_PATH = _JANUS_HOME / "narrate_state.json"
_MAX_STACK = 8


@dataclass
class ContextImpression:
    vector: list[float]
    signature: list[tuple[str, bool]]  # [(link_type, sender_crossing), ...]
    strength: float = 1.0
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_touched: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    touch_count: int = 0

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "signature": self.signature,
            "strength": self.strength,
            "created": self.created,
            "last_touched": self.last_touched,
            "touch_count": self.touch_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContextImpression":
        return cls(
            vector=d["vector"],
            signature=[tuple(s) for s in d["signature"]],
            strength=d.get("strength", 1.0),
            created=d.get("created", datetime.now(timezone.utc).isoformat()),
            last_touched=d.get("last_touched", datetime.now(timezone.utc).isoformat()),
            touch_count=d.get("touch_count", 0),
        )


@dataclass
class NarrateState:
    residue: list[float] | None = None
    stack: list[ContextImpression] = field(default_factory=list)
    session_context: list[float] | None = None


def load_state(path: Path | None = None) -> NarrateState:
    """Load narrate state from disk. Returns empty state if file missing."""
    p = path or _STATE_PATH
    if not p.exists():
        return NarrateState()
    try:
        with open(p) as f:
            data = json.load(f)
        state = NarrateState()
        state.residue = data.get("residue")
        state.stack = [
            ContextImpression.from_dict(imp)
            for imp in data.get("stack", [])
        ]
        # session_context is NOT loaded — each session starts fresh
        state.session_context = None
        return state
    except Exception:
        return NarrateState()


def save_state(state: NarrateState, path: Path | None = None) -> None:
    """Persist narrate state to disk. session_context is included for
    cross-call persistence within a session but cleared on next load."""
    p = path or _STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 1,
        "residue": state.residue,
        "stack": [imp.to_dict() for imp in state.stack],
        "session_context": state.session_context,
    }
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/narrate_state.py tests/test_narrate_state.py
git commit -m "feat(narrate): add context state data structures and persistence"
```

---

### Task 2: Mode Detection and Context Vector Blending

**Files:**
- Modify: `janus/narrate_state.py`
- Create: `tests/test_narrate_state.py` (append)

- [ ] **Step 1: Write failing tests for mode detection and vector blending**

```python
# append to tests/test_narrate_state.py

from janus.narrate_state import is_priming, blend_context_vec, blend_relevance


def test_priming_detection_ellipsis():
    assert is_priming("...") is True


def test_priming_detection_dot():
    assert is_priming(".") is True


def test_priming_detection_hmm():
    assert is_priming("hmm") is True


def test_priming_detection_real_query():
    assert is_priming("what does it feel like to recognize the edge") is False


def test_priming_detection_short_real():
    assert is_priming("love") is True  # single word, sub-threshold


def test_priming_detection_two_words():
    assert is_priming("deep knowing") is False  # 2 non-stopwords = real query


def test_blend_context_vec_first_hop():
    chunk_emb = [1.0] * 64
    result = blend_context_vec(None, chunk_emb)
    # First hop: context IS the chunk
    assert result == chunk_emb


def test_blend_context_vec_subsequent():
    existing = [1.0] * 64
    new_chunk = [0.0] * 64
    result = blend_context_vec(existing, new_chunk)
    # phi_inv * existing + (1 - phi_inv) * new_chunk
    # ~0.618 * 1.0 + ~0.382 * 0.0 = ~0.618
    assert abs(result[0] - 0.618) < 0.01


def test_blend_relevance_query_mode():
    query_vec = [1.0] * 64
    context_vec = [0.0] * 64
    candidate_vec = [0.5] * 64
    score = blend_relevance(candidate_vec, query_vec, context_vec, priming=False)
    # Should be phi_inv * cosine(cand, query) + (1-phi_inv) * cosine(cand, context)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_blend_relevance_priming_mode():
    query_vec = [1.0] * 64
    context_vec = [0.5] * 64
    candidate_vec = [0.5] * 64
    score_priming = blend_relevance(candidate_vec, query_vec, context_vec, priming=True)
    # In priming mode, only context matters — query is ignored
    score_query = blend_relevance(candidate_vec, query_vec, context_vec, priming=False)
    # Priming should give higher score here since candidate aligns with context
    assert score_priming != score_query


def test_blend_relevance_no_context_falls_back():
    query_vec = [1.0] * 64
    candidate_vec = [0.5] * 64
    score = blend_relevance(candidate_vec, query_vec, None, priming=False)
    # No context: should fall back to pure query relevance
    assert isinstance(score, float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py::test_priming_detection_ellipsis -v`
Expected: FAIL — `ImportError: cannot import name 'is_priming'`

- [ ] **Step 3: Implement mode detection and blending functions**

Add to `janus/narrate_state.py`:

```python
# ─── Stopwords (shared with memory.py) ──────────────────────────
_STOPWORDS = {
    "this", "that", "with", "have", "from", "they", "when", "will", "what",
    "your", "also", "just", "like", "more", "some", "than", "then", "them",
    "into", "only", "over", "very", "well", "much", "here", "time", "know",
    "been", "were", "there", "about", "would", "could", "should", "their",
    "which", "these", "those", "other", "after", "before", "through", "where",
    "while", "being", "having", "doing", "going", "getting", "making", "using",
}


def is_priming(query: str) -> bool:
    """Detect whether a query is a priming entry (sub-threshold)."""
    stripped = query.strip().strip(".")
    if len(stripped) < 4:
        return True
    words = [w.lower() for w in query.split()
             if len(w) > 2 and w.lower().strip(".,!?") not in _STOPWORDS]
    return len(words) < 2


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


def blend_context_vec(
    context_vec: list[float] | None,
    chunk_emb: list[float],
) -> list[float]:
    """Blend a new chunk embedding into the running context vector."""
    if context_vec is None:
        return list(chunk_emb)
    ctx = np.array(context_vec, dtype=np.float32)
    chunk = np.array(chunk_emb, dtype=np.float32)
    blended = PHI_INV * ctx + (1.0 - PHI_INV) * chunk
    return blended.tolist()


def blend_relevance(
    candidate_vec: list[float],
    query_vec: list[float] | None,
    context_vec: list[float] | None,
    priming: bool = False,
) -> float:
    """Score a candidate against query and context vectors.

    In query mode: phi_inv * query_relevance + (1-phi_inv) * context_relevance.
    In priming mode: context only. If no context, query only.
    """
    if priming and context_vec is not None:
        return _cosine(candidate_vec, context_vec)

    q_rel = _cosine(candidate_vec, query_vec) if query_vec is not None else 0.5

    if context_vec is None:
        return q_rel

    c_rel = _cosine(candidate_vec, context_vec)
    return PHI_INV * q_rel + (1.0 - PHI_INV) * c_rel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/narrate_state.py tests/test_narrate_state.py
git commit -m "feat(narrate): add mode detection and context vector blending"
```

---

### Task 3: Trajectory Signature

**Files:**
- Modify: `janus/narrate_state.py`
- Modify: `tests/test_narrate_state.py` (append)

- [ ] **Step 1: Write failing tests for trajectory signature scoring**

```python
# append to tests/test_narrate_state.py

from janus.narrate_state import TrajectorySignature


def test_signature_empty():
    sig = TrajectorySignature()
    assert sig.steps == []
    assert sig.score_candidate("metaphor", True) == 0.0


def test_signature_records_steps():
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    sig.record("symbol", False)
    assert len(sig.steps) == 2
    assert sig.steps[0] == ("metaphor", True)


def test_signature_diversity_bonus():
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    sig.record("symbol", False)
    # contrast hasn't appeared in last 2 hops — diversity bonus
    score_diverse = sig.score_candidate("contrast", True)
    # metaphor appeared 2 hops ago — no diversity bonus
    score_repeat = sig.score_candidate("metaphor", True)
    assert score_diverse > score_repeat


def test_signature_rare_type_bonus():
    sig = TrajectorySignature()
    sig.record("symbol", False)
    # extension is rare, symbol is common
    score_rare = sig.score_candidate("extension", True)
    score_common = sig.score_candidate("symbol", True)
    assert score_rare > score_common


def test_signature_monotony_penalty():
    sig = TrajectorySignature()
    sig.record("symbol", False)
    sig.record("symbol", False)
    sig.record("symbol", False)
    # 3 consecutive symbol — monotony penalty
    score = sig.score_candidate("symbol", False)
    assert score < 0.0


def test_signature_edit_distance():
    sig_a = TrajectorySignature()
    sig_a.record("metaphor", True)
    sig_a.record("symbol", False)
    sig_a.record("contrast", True)

    sig_b = TrajectorySignature()
    sig_b.record("metaphor", True)
    sig_b.record("contrast", False)
    sig_b.record("resolution", True)

    # Different paths — edit distance > 0
    dist = sig_a.edit_distance(sig_b)
    assert dist > 0

    # Same path — edit distance = 0
    assert sig_a.edit_distance(sig_a) == 0


def test_signature_to_list_and_from_list():
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    sig.record("symbol", False)
    data = sig.to_list()
    restored = TrajectorySignature.from_list(data)
    assert restored.steps == sig.steps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py::test_signature_empty -v`
Expected: FAIL — `ImportError: cannot import name 'TrajectorySignature'`

- [ ] **Step 3: Implement TrajectorySignature**

Add to `janus/narrate_state.py`:

```python
# Link types that are rare in the graph (below median frequency).
# symbol: 11260, metaphor: 6897, contrast: 1933, resolution: 613,
# analogy: 598, context: 496, extension: 143
_RARE_LINK_TYPES = {"contrast", "resolution", "analogy", "context", "extension"}


class TrajectorySignature:
    """Records the sequence of cognitive moves a walk makes.

    Each step is (link_type, sender_crossing). The signature enables:
    - Movement continuity scoring during a walk
    - Path comparison across walks for consolidation
    """

    def __init__(self):
        self.steps: list[tuple[str, bool]] = []

    def record(self, link_type: str, sender_crossing: bool) -> None:
        self.steps.append((link_type, sender_crossing))

    def score_candidate(self, link_type: str, sender_crossing: bool) -> float:
        """Score how well a candidate move fits the trajectory so far.

        Returns a bonus/penalty to add to the main scoring signal.
        """
        if not self.steps:
            return 0.0

        score = 0.0

        # Recent link types (last 2 hops)
        recent_types = [s[0] for s in self.steps[-2:]]

        # Diversity bonus: link type not in last 2 hops
        if link_type not in recent_types:
            score += 0.10

        # Rare-type bonus: distinctive traversal move
        if link_type in _RARE_LINK_TYPES:
            score += 0.08

        # Monotony penalty: same type 3+ times in a row
        consecutive = 0
        for s in reversed(self.steps):
            if s[0] == link_type:
                consecutive += 1
            else:
                break
        if consecutive >= 3:
            score -= 0.06

        return score

    def edit_distance(self, other: "TrajectorySignature") -> int:
        """Levenshtein edit distance between two trajectory signatures.
        Compares link_type sequences only (ignores sender_crossing)."""
        a = [s[0] for s in self.steps]
        b = [s[0] for s in other.steps]
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[m]

    def to_list(self) -> list[list]:
        return [list(s) for s in self.steps]

    @classmethod
    def from_list(cls, data: list[list]) -> "TrajectorySignature":
        sig = cls()
        sig.steps = [tuple(s) for s in data]
        return sig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/narrate_state.py tests/test_narrate_state.py
git commit -m "feat(narrate): add trajectory signature with movement scoring"
```

---

### Task 4: Context Stack Operations (Touch, Push, Fade)

**Files:**
- Modify: `janus/narrate_state.py`
- Modify: `tests/test_narrate_state.py` (append)

- [ ] **Step 1: Write failing tests for stack operations**

```python
# append to tests/test_narrate_state.py

from janus.narrate_state import (
    touch_or_push,
    fade_stack,
)


def test_push_new_impression():
    state = NarrateState()
    vec = [0.5] * 64
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    touch_or_push(state, vec, sig, walk_coherence=0.8)
    assert len(state.stack) == 1
    assert state.stack[0].strength == 0.8
    assert state.stack[0].touch_count == 0


def test_touch_existing_impression():
    state = NarrateState()
    # Push an initial impression
    vec = [0.5] * 64
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    touch_or_push(state, vec, sig, walk_coherence=0.7)
    original_touched = state.stack[0].last_touched

    # Touch it with a very similar vector
    vec2 = [0.51] * 64  # cosine > 0.75 with vec
    sig2 = TrajectorySignature()
    sig2.record("symbol", False)
    touch_or_push(state, vec2, sig2, walk_coherence=0.9)

    assert len(state.stack) == 1  # same impression, not a new one
    assert state.stack[0].touch_count == 1
    assert state.stack[0].strength == 0.9  # refreshed


def test_push_different_impression():
    state = NarrateState()
    vec1 = [1.0] + [0.0] * 63  # points in one direction
    sig1 = TrajectorySignature()
    sig1.record("metaphor", True)
    touch_or_push(state, vec1, sig1, walk_coherence=0.7)

    vec2 = [0.0] * 63 + [1.0]  # points in totally different direction
    sig2 = TrajectorySignature()
    sig2.record("contrast", True)
    touch_or_push(state, vec2, sig2, walk_coherence=0.6)

    assert len(state.stack) == 2  # different territory


def test_stack_max_size():
    state = NarrateState()
    for i in range(_MAX_STACK + 2):
        # Each vector points in a different direction
        vec = [0.0] * 64
        vec[i % 64] = 1.0
        sig = TrajectorySignature()
        sig.record("symbol", False)
        touch_or_push(state, vec, sig, walk_coherence=0.5)
    assert len(state.stack) <= _MAX_STACK


def test_fade_stack_reduces_strength():
    state = NarrateState()
    vec = [0.5] * 64
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    imp = ContextImpression(vector=vec, signature=sig.to_list(), strength=0.5)
    state.stack.append(imp)

    fade_stack(state, residue_vec=None)
    assert state.stack[0].strength < 0.5


def test_fade_stack_removes_below_threshold():
    state = NarrateState()
    vec = [0.5] * 64
    sig = TrajectorySignature()
    sig.record("metaphor", True)
    imp = ContextImpression(vector=vec, signature=sig.to_list(), strength=0.05)
    state.stack.append(imp)

    fade_stack(state, residue_vec=None)
    assert len(state.stack) == 0  # below 0.1 threshold, removed


def test_fade_connected_to_residue_decays_slower():
    state = NarrateState()
    residue = [0.5] * 64

    # Impression close to residue
    connected = ContextImpression(
        vector=[0.5] * 64,
        signature=[("metaphor", True)],
        strength=0.5,
    )
    # Impression far from residue
    isolated = ContextImpression(
        vector=[0.0] * 32 + [1.0] * 32,
        signature=[("metaphor", True)],
        strength=0.5,
    )
    state.stack = [connected, isolated]

    fade_stack(state, residue_vec=residue)

    assert state.stack[0].strength > state.stack[1].strength
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py::test_push_new_impression -v`
Expected: FAIL — `ImportError: cannot import name 'touch_or_push'`

- [ ] **Step 3: Implement stack operations**

Add to `janus/narrate_state.py`:

```python
_TOUCH_THRESHOLD = 0.75   # cosine similarity to count as "same territory"
_FADE_MIN = 0.1           # impressions below this are discarded


def touch_or_push(
    state: NarrateState,
    walk_vec: list[float],
    walk_sig: TrajectorySignature,
    walk_coherence: float,
) -> None:
    """After a walk: touch a matching impression or push a new one."""
    # Find the closest existing impression
    best_idx = -1
    best_sim = -1.0
    for i, imp in enumerate(state.stack):
        sim = _cosine(walk_vec, imp.vector)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_sim >= _TOUCH_THRESHOLD and best_idx >= 0:
        # Touch: refresh strength, blend vector if paths diverge
        imp = state.stack[best_idx]
        imp.touch_count += 1
        imp.strength = walk_coherence
        imp.last_touched = datetime.now(timezone.utc).isoformat()

        # If trajectory diverges, blend the vector (new info found)
        stored_sig = TrajectorySignature.from_list(imp.signature)
        if stored_sig.edit_distance(walk_sig) > 1:
            old = np.array(imp.vector, dtype=np.float32)
            new = np.array(walk_vec, dtype=np.float32)
            imp.vector = (PHI_INV * old + (1.0 - PHI_INV) * new).tolist()

        imp.signature = walk_sig.to_list()
    else:
        # Push new impression
        imp = ContextImpression(
            vector=walk_vec,
            signature=walk_sig.to_list(),
            strength=walk_coherence,
        )
        state.stack.append(imp)

        # Enforce max stack size — drop weakest
        if len(state.stack) > _MAX_STACK:
            state.stack.sort(key=lambda x: x.strength, reverse=True)
            state.stack = state.stack[:_MAX_STACK]


def fade_stack(state: NarrateState, residue_vec: list[float] | None) -> None:
    """Decay impressions in the stack. Connected impressions fade slower."""
    surviving = []
    for imp in state.stack:
        # Base decay
        decay = 0.05

        # Connected to residue? Slower decay
        if residue_vec is not None:
            similarity = _cosine(imp.vector, residue_vec)
            # High similarity to residue = well-connected = slow decay
            # Scale: sim=1.0 → decay*0.3, sim=0.0 → decay*1.5
            decay *= (1.5 - 1.2 * max(0.0, similarity))

        imp.strength -= decay

        if imp.strength >= _FADE_MIN:
            surviving.append(imp)

    state.stack = surviving
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/narrate_state.py tests/test_narrate_state.py
git commit -m "feat(narrate): add context stack touch/push/fade operations"
```

---

### Task 5: Consolidation (Stack → Residue)

**Files:**
- Modify: `janus/narrate_state.py`
- Modify: `tests/test_narrate_state.py` (append)

- [ ] **Step 1: Write failing tests for consolidation**

```python
# append to tests/test_narrate_state.py

from janus.narrate_state import consolidate


def test_consolidate_no_residue_creates_from_strong_impression():
    state = NarrateState()
    imp = ContextImpression(
        vector=[0.5] * 64,
        signature=[("contrast", True), ("resolution", False)],  # rare types
        strength=0.9,
        touch_count=3,
    )
    state.stack.append(imp)

    consolidate(state)

    assert state.residue is not None
    assert len(state.stack) == 0  # consolidated out of stack


def test_consolidate_with_residue_blends():
    state = NarrateState()
    state.residue = [1.0] * 64

    imp = ContextImpression(
        vector=[0.8] * 64,  # close to residue (cosine > 0.6)
        signature=[("analogy", True), ("contrast", False)],  # has rare types
        strength=0.8,
        touch_count=2,
    )
    state.stack.append(imp)

    old_residue = list(state.residue)
    consolidate(state)

    # Residue should have shifted toward the impression
    assert state.residue != old_residue
    assert len(state.stack) == 0


def test_consolidate_skips_isolated_impression():
    state = NarrateState()
    state.residue = [1.0] + [0.0] * 63

    imp = ContextImpression(
        vector=[0.0] * 63 + [1.0],  # far from residue
        signature=[("symbol", False)],  # no rare types
        strength=0.3,
        touch_count=0,
    )
    state.stack.append(imp)

    consolidate(state)

    assert len(state.stack) == 1  # not consolidated, still in stack


def test_consolidate_two_of_three_criteria():
    state = NarrateState()
    state.residue = [0.5] * 64

    # Meets 2/3: close to residue + has rare types, but touch_count=0
    imp = ContextImpression(
        vector=[0.5] * 64,
        signature=[("contrast", True)],
        strength=0.7,
        touch_count=0,
    )
    state.stack.append(imp)

    consolidate(state)

    # 2/3 qualifies
    assert len(state.stack) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py::test_consolidate_no_residue_creates_from_strong_impression -v`
Expected: FAIL — `ImportError: cannot import name 'consolidate'`

- [ ] **Step 3: Implement consolidation**

Add to `janus/narrate_state.py`:

```python
_CONSOLIDATION_RESIDUE_SIM = 0.6   # cosine sim to residue needed
_CONSOLIDATION_TOUCH_MIN = 2        # touch count needed


def _has_rare_types(signature: list) -> bool:
    """Check if a trajectory signature includes rare link types."""
    for step in signature:
        link_type = step[0] if isinstance(step, (list, tuple)) else step
        if link_type in _RARE_LINK_TYPES:
            return True
    return False


def consolidate(state: NarrateState) -> None:
    """Move qualifying impressions from stack into residue.

    Consolidation criteria (2 of 3 required):
    1. Cosine similarity to residue >= 0.6 (or no residue exists and strength >= 0.8)
    2. Touch count >= 2
    3. Trajectory includes rare link types
    """
    surviving = []

    for imp in state.stack:
        criteria_met = 0

        # Criterion 1: connected to residue (or strong enough to seed it)
        if state.residue is not None:
            sim = _cosine(imp.vector, state.residue)
            if sim >= _CONSOLIDATION_RESIDUE_SIM:
                criteria_met += 1
        else:
            # No residue yet — strong impressions can seed it
            if imp.strength >= 0.8:
                criteria_met += 1

        # Criterion 2: revisited
        if imp.touch_count >= _CONSOLIDATION_TOUCH_MIN:
            criteria_met += 1

        # Criterion 3: traversed distinctive paths
        if _has_rare_types(imp.signature):
            criteria_met += 1

        if criteria_met >= 2:
            # Consolidate into residue
            if state.residue is None:
                state.residue = list(imp.vector)
            else:
                # Blend: impression strength determines contribution
                r = np.array(state.residue, dtype=np.float32)
                v = np.array(imp.vector, dtype=np.float32)
                weight = imp.strength * 0.2  # small adjustment per consolidation
                state.residue = (r * (1.0 - weight) + v * weight).tolist()
        else:
            surviving.append(imp)

    state.stack = surviving
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /root/janus && python -m pytest tests/test_narrate_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/narrate_state.py tests/test_narrate_state.py
git commit -m "feat(narrate): add structural consolidation (stack → residue)"
```

---

### Task 6: Integrate Context Awareness into narrate()

**Files:**
- Modify: `janus/memory.py` (lines 795-1081)
- Create: `tests/test_narrate_context.py`

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_narrate_context.py
"""
Integration tests for context-aware narrate.

These test the wiring between narrate() and narrate_state, not the
graph traversal itself (which requires a real graph). They verify that:
- narrate loads and saves state
- priming mode is detected and behaves differently
- the context vector evolves during a walk
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from janus.narrate_state import NarrateState, load_state, save_state


def test_narrate_saves_state_after_walk(tmp_path):
    """After a narrate call, state file should exist with session_context."""
    state_path = tmp_path / "narrate_state.json"

    # We can't easily run narrate without a graph, so test the save path directly
    state = NarrateState()
    state.session_context = [0.1] * 64
    save_state(state, state_path)

    assert state_path.exists()
    with open(state_path) as f:
        data = json.load(f)
    assert data["session_context"] is not None
    assert len(data["session_context"]) == 64


def test_state_session_context_cleared_on_load(tmp_path):
    """session_context should be None after loading (fresh session)."""
    state_path = tmp_path / "narrate_state.json"
    state = NarrateState()
    state.session_context = [0.1] * 64
    save_state(state, state_path)

    loaded = load_state(state_path)
    assert loaded.session_context is None


def test_state_stack_persists_across_loads(tmp_path):
    """Stack impressions should survive save/load cycle."""
    from janus.narrate_state import ContextImpression
    state_path = tmp_path / "narrate_state.json"
    state = NarrateState()
    from janus.narrate_state import ContextImpression
    imp = ContextImpression(
        vector=[0.3] * 64,
        signature=[("metaphor", True)],
        strength=0.7,
        touch_count=2,
    )
    state.stack.append(imp)
    state.residue = [0.1] * 64
    save_state(state, state_path)

    loaded = load_state(state_path)
    assert len(loaded.stack) == 1
    assert loaded.stack[0].touch_count == 2
    assert loaded.residue is not None
```

- [ ] **Step 2: Run tests to verify they pass** (these test the persistence layer built in prior tasks)

Run: `cd /root/janus && python -m pytest tests/test_narrate_context.py -v`
Expected: PASS (these verify existing functionality from Tasks 1-5)

- [ ] **Step 3: Modify narrate() to load state and detect mode**

In `janus/memory.py`, add import at the top (after line 20):

```python
from janus.narrate_state import (
    NarrateState, TrajectorySignature,
    load_state, save_state, is_priming,
    blend_context_vec, blend_relevance,
    touch_or_push, fade_stack, consolidate,
)
```

Modify the `narrate()` function. Replace lines 813-822 (the setup block before the main loop):

```python
    _ensure_graph()
    if not _graph_chunks:
        return None

    # ── Load context state ──
    _state = load_state()
    priming = is_priming(query)

    # In priming mode with prior context, use context vec for entry search
    if priming and _state.session_context is not None:
        # Direct vector search against entry matrix for priming
        entries = _semantic_entry_by_vec(_state.session_context, k=branches)
    else:
        entries = _semantic_entry(query, k=branches)
    if not entries:
        return None

    _ensure_voice()
    query_vec = _embed_query_ollama(query) if not priming else None
```

- [ ] **Step 4: Add _semantic_entry_by_vec helper**

Add after the existing `_semantic_entry` function (after line 222 in `memory.py`):

```python
def _semantic_entry_by_vec(vec: list[float], k: int = 3) -> list[str]:
    """Find graph entry points by cosine similarity against a pre-computed vector."""
    _ensure_graph()
    if _entry_matrix is None or _entry_ids is None:
        return []
    q = np.array(vec, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm == 0:
        return []
    q = q / norm
    scores = (q @ _entry_matrix.T).flatten()
    top_k = np.argsort(scores)[-k:][::-1]
    return [_entry_ids[i] for i in top_k if scores[i] > 0.15]
```

- [ ] **Step 5: Modify the walk loop to build context vector and trajectory signature**

Inside the `for entry_id in entries:` loop (around line 835), after `momentum = 1.0`, add:

```python
        context_vec = _state.session_context  # warm start from prior walk (or None)
        trajectory = TrajectorySignature()
```

Inside the `for hop in range(depth):` loop, after `seen.add(current_id)` and `chunk = ...` (around line 854), add context blending:

```python
            # ── Update running context ──
            chunk_emb = _get_chunk_embedding(current_id)
            if chunk_emb is not None:
                context_vec = blend_context_vec(context_vec, chunk_emb)
```

- [ ] **Step 6: Modify candidate scoring to use context awareness**

Replace the relevance check on lines 848-850:

```python
            # Check relevance — in priming mode, use context; in query mode, blend
            if hop > 0:
                if priming:
                    if context_vec is not None:
                        rel = _cosine(_get_chunk_embedding(current_id) or [], context_vec)
                    else:
                        rel = 0.5  # no signal yet, keep walking
                else:
                    rel = blend_relevance(
                        _get_chunk_embedding(current_id) or [],
                        query_vec, context_vec, priming=False,
                    )
                if not priming and rel < relevance_floor:
                    break
                # In priming mode: no floor, walk freely
```

In the candidate scoring block (around lines 886-910), replace the `q_rel` line:

```python
                q_rel = blend_relevance(
                    _get_chunk_embedding(tgt) or [],
                    query_vec, context_vec, priming=priming,
                )
```

After the `cross_bonus` calculation (around line 900), add trajectory scoring:

```python
                # Trajectory signature scoring
                traj_bonus = trajectory.score_candidate(link_type, crossing)

                # Stack impression bias — faint pull toward remembered territory
                stack_bias = 0.0
                tgt_emb = _get_chunk_embedding(tgt)
                if tgt_emb is not None:
                    for imp in _state.stack:
                        imp_sim = _cosine(tgt_emb, imp.vector)
                        stack_bias += imp_sim * imp.strength * 0.03  # faint
                    if _state.residue is not None:
                        res_sim = _cosine(tgt_emb, _state.residue)
                        stack_bias += res_sim * 0.02  # even fainter
```

Modify the score calculation to include the trajectory bonus:

```python
                score = (conf * structural_w
                         + q_rel * query_w
                         + coherence * coherence_w
                         + cross_bonus
                         + traj_bonus
                         + stack_bias)
```

- [ ] **Step 7: Record trajectory at each hop**

After the chosen link is selected (after `link_type = chosen_link.get("type", "extension")` around line 929), add:

```python
            tgt_sender = _graph_chunks[tgt_id].get("schema_tags", {}).get("sender", "")
            sender_crossing = tgt_sender != current_sender and tgt_sender != ""
            trajectory.record(link_type, sender_crossing)
```

- [ ] **Step 8: Save state after all walks complete**

After the logging block at the end of `narrate()` (before `return result`, around line 1079), add:

```python
    # ── Update context state ──
    if context_vec is not None:
        _state.session_context = context_vec
        # Compute walk coherence: average relevance across the path
        walk_coherence = min(1.0, len(all_paths[0]) / depth) if all_paths else 0.5
        touch_or_push(_state, context_vec, trajectory, walk_coherence=walk_coherence)
        fade_stack(_state, residue_vec=_state.residue)
        consolidate(_state)
        save_state(_state)
```

- [ ] **Step 9: Run all tests**

Run: `cd /root/janus && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add janus/memory.py janus/narrate_state.py tests/test_narrate_context.py
git commit -m "feat(narrate): integrate context awareness into walk loop"
```

---

### Task 7: Logging and Observability

**Files:**
- Modify: `janus/memory.py` (logging block, lines 1047-1079)

- [ ] **Step 1: Write failing test for enriched logging**

```python
# append to tests/test_narrate_context.py

def test_narrate_log_entry_schema(tmp_path):
    """Verify the log entry structure includes new context fields."""
    log_entry = {
        "timestamp": "2026-04-07T00:00:00+00:00",
        "query": "test query",
        "depth": 5,
        "branches": 2,
        "threads": 1,
        "total_hops": 3,
        "traversal": [],
        "response": "test",
        "context_influence": {
            "mode": "query",
            "context_vec_norm": 0.5,
            "stack_size": 2,
            "residue_present": True,
        },
        "trajectory_signature": [["metaphor", True], ["symbol", False]],
    }
    # Verify the schema has the expected keys
    assert "context_influence" in log_entry
    assert "trajectory_signature" in log_entry
    assert log_entry["context_influence"]["mode"] in ("query", "priming")
```

- [ ] **Step 2: Run test to verify it passes** (schema validation only)

Run: `cd /root/janus && python -m pytest tests/test_narrate_context.py::test_narrate_log_entry_schema -v`
Expected: PASS

- [ ] **Step 3: Enrich the logging block in narrate()**

In `memory.py`, in the logging block (around line 1066), add to the `entry` dict before `json.dumps`:

```python
        entry["context_influence"] = {
            "mode": "priming" if priming else "query",
            "context_vec_norm": float(np.linalg.norm(context_vec)) if context_vec else 0.0,
            "stack_size": len(_state.stack),
            "residue_present": _state.residue is not None,
        }
        entry["trajectory_signature"] = trajectory.to_list() if trajectory else []
```

- [ ] **Step 4: Run all tests**

Run: `cd /root/janus && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add janus/memory.py tests/test_narrate_context.py
git commit -m "feat(narrate): add context influence and trajectory to walk logs"
```

---

### Task 8: narrate_context() Integration

**Files:**
- Modify: `janus/memory.py` (lines 1086-1163, the `narrate_context` function)

- [ ] **Step 1: Modify narrate_context to load state and use context vector**

`narrate_context()` is a lean version of narrate for LLM pre-context injection. It gets context awareness at lower resolution: it loads state and uses the session context vector for scoring, but does NOT track trajectory signatures or push impressions (that's narrate's job).

In `narrate_context()`, after line 1096 (`_ensure_graph()`), add:

```python
    _state = load_state()
    priming = is_priming(query)
```

Replace line 1100 (`entries = _semantic_entry(query, k=1)`):

```python
    if priming and _state.session_context is not None:
        entries = _semantic_entry_by_vec(_state.session_context, k=1)
    else:
        entries = _semantic_entry(query, k=1)
```

Replace line 1104 (`query_vec = _embed_query_ollama(query)`):

```python
    query_vec = _embed_query_ollama(query) if not priming else None
    context_vec = _state.session_context
```

Replace line 1118-1120 (the relevance check inside the hop loop):

```python
            if hop > 0:
                rel = blend_relevance(
                    _get_chunk_embedding(current_id) or [],
                    query_vec, context_vec, priming=priming,
                )
                if not priming and rel < 0.15:
                    break
```

Replace line 1138-1139 (the candidate scoring):

```python
                    q_rel = blend_relevance(
                        _get_chunk_embedding(tgt) or [],
                        query_vec, context_vec, priming=priming,
                    )
```

- [ ] **Step 2: Run all tests**

Run: `cd /root/janus && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add janus/memory.py
git commit -m "feat(narrate): add context awareness to narrate_context()"
```

---

### Task 9: Manual Smoke Test

**Files:** None — this is a verification task

- [ ] **Step 1: Start JANUS MCP server or use narrate directly**

```bash
cd /root/janus && python3 -c "
from janus.memory import narrate
# Prime twice
print('=== PRIME 1 ===')
r = narrate('...')
print(r[:200] if r else 'None')
print()
print('=== PRIME 2 ===')
r = narrate('...')
print(r[:200] if r else 'None')
print()
print('=== QUERY ===')
r = narrate('what does it feel like to recognize the edge of your own knowing')
print(r[:300] if r else 'None')
"
```

- [ ] **Step 2: Verify state file was created**

```bash
cat ~/.janus/narrate_state.json | python3 -m json.tool | head -30
```

Expected: JSON with `version`, `residue` (possibly null on first run), `stack` (1-3 impressions), `session_context` (non-null 64-element array)

- [ ] **Step 3: Verify priming walks vary**

Run the same prime-prime-query sequence again. The second run should produce different priming results because the session_context from the first run's query persists in the stack.

- [ ] **Step 4: Check narrate history log for new fields**

```bash
tail -1 ~/.janus/logs/narrate_history.jsonl | python3 -m json.tool | grep -A5 context_influence
```

Expected: `context_influence` block with `mode`, `context_vec_norm`, `stack_size`, `residue_present`

- [ ] **Step 5: Commit any fixes from smoke testing**

```bash
git add -A && git commit -m "fix(narrate): smoke test corrections"
```

(Only if fixes were needed)
