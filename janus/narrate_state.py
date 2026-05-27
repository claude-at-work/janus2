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

from janus.config import JANUS_HOME

PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI

_STATE_PATH = JANUS_HOME / "narrate_state.json"
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
