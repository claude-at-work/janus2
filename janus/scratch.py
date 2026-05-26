"""
JANUS Scratch Pad -- lightweight context preservation.

L1: in-memory scratch_pad (live session, managed by shell.py)
L2: JSON journal (~/.janus/scratch.json) -- persistent, cross-session
L3: bud semantic index -- long-term recall via embeddings (graceful fallback)

Criticality levels:
- critical: survives compression, persists longer
- important: moderate persistence
- normal: expires with normal flow
"""

import json
import os
import uuid
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from janus.memory import JANUS_HOME, maybe_compress
from janus.llm import summarize

SCRATCH_FILE = JANUS_HOME / "scratch.json"
MAX_SCRATCH_ITEMS = 100        # max items in L2 journal
CRITICAL_PRESERVE_THRESHOLD = 60  # critical items survive compression if under this age
NORMAL_EXPIRE_THRESHOLD = 40    # normal items expire after this many turns


def _ensure_dirs():
    JANUS_HOME.mkdir(exist_ok=True)


@dataclass
class ScratchItem:
    """A single scratch pad entry."""
    id: str
    content: str
    created: str
    tags: List[str]
    criticality: str  # 'critical', 'important', 'normal'
    checked: bool = False
    expires_at: Optional[int] = None  # turn count when item expires

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ScratchItem':
        return cls(
            id=d['id'],
            content=d['content'],
            created=d['created'],
            tags=d.get('tags', []),
            criticality=d.get('criticality', 'normal'),
            checked=d.get('checked', False),
            expires_at=d.get('expires_at')
        )


# ─── L1: In-memory scratch pad ──────────────────────────────────

class ScratchPad:
    """
    In-memory scratch pad for current session.

    Manages temporary context that would otherwise be lost in conversation
    flow. Items can be checked off when complete.
    """

    def __init__(self):
        self._items: List[ScratchItem] = []
        self._turn_count = 0
        self._session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, content: str, tags: List[str] = None,
            criticality: str = 'normal', expires_after: int = None) -> ScratchItem:
        """
        Add a new scratch item.

        Args:
            content: The note content
            tags: Optional labels for filtering
            criticality: 'critical', 'important', or 'normal'
            expires_after: Turn count until expiration (None = use default)

        Returns:
            The created ScratchItem
        """
        self._turn_count += 1

        item = ScratchItem(
            id=str(uuid.uuid4())[:8],
            content=content,
            created=datetime.datetime.now().isoformat(),
            tags=tags or [],
            criticality=criticality or 'normal',
            checked=False,
            expires_at=expires_after or self._turn_count + NORMAL_EXPIRE_THRESHOLD
        )
        self._items.append(item)
        return item

    def get(self, item_id: str) -> Optional[ScratchItem]:
        """Get an item by ID."""
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def list(self, checked: Optional[bool] = None) -> List[ScratchItem]:
        """List items, optionally filtered by checked status."""
        if checked is None:
            return self._items[:]
        return [i for i in self._items if i.checked == checked]

    def filter(self, tags: List[str] = None,
               criticality: str = None) -> List[ScratchItem]:
        """Filter items by tags and/or criticality."""
        result = self._items[:]
        if tags:
            result = [i for i in result if any(t in tags for t in i.tags)]
        if criticality:
            result = [i for i in result if i.criticality == criticality]
        return result

    def complete(self, item_id: str) -> bool:
        """Mark an item as complete (checked)."""
        item = self.get(item_id)
        if item:
            item.checked = True
            return True
        return False

    def remove(self, item_id: str) -> bool:
        """Remove an item from the pad."""
        for i, item in enumerate(self._items):
            if item.id == item_id:
                del self._items[i]
                return True
        return False

    def clear(self):
        """Clear all items."""
        self._items = []
        self._turn_count = 0

    def to_dicts(self) -> List[dict]:
        """Convert all items to dictionaries."""
        return [i.to_dict() for i in self._items]

    def active_count(self) -> int:
        """Count of un-checked items."""
        return sum(1 for i in self._items if not i.checked)

    def update_turn_count(self, count: int):
        """Update current turn count for expiration calculations."""
        self._turn_count = count


# ─── L2: Persistent scratch pad ─────────────────────────────────

def load_scratch() -> List[dict]:
    """
    Load scratch pad items from L2 JSON journal.

    Returns items from last session, up to MAX_SCRATCH_ITEMS.
    Gracefully returns [] if file doesn't exist.
    """
    _ensure_dirs()
    try:
        if SCRATCH_FILE.exists():
            with open(SCRATCH_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data[-MAX_SCRATCH_ITEMS:]
        return []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_scratch(items: List[dict]):
    """
    Persist scratch pad items to L2 JSON journal.

    Capped at MAX_SCRATCH_ITEMS for token efficiency.
    Graceful fallback: doesn't fail if write fails.
    """
    _ensure_dirs()
    try:
        existing = []
        if SCRATCH_FILE.exists():
            try:
                with open(SCRATCH_FILE) as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing = []

        combined = existing + items
        trimmed = combined[-MAX_SCRATCH_ITEMS:]

        # Write atomically via temp file
        temp_file = SCRATCH_FILE.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(trimmed, f, indent=2)
        temp_file.rename(SCRATCH_FILE)
    except Exception:
        pass  # Graceful degradation - no scratch is better than crashing


def expire_old_scratch(turn_count: int = None) -> List[dict]:
    """
    Remove expired and checked items from scratch pad.

    Criticality-based expiration:
    - critical: survives if under CRITICAL_PRESERVE_THRESHOLD turns old
    - important: survives for moderate turns
    - normal: expires after NORMAL_EXPIRE_THRESHOLD

    Returns list of remaining items.
    """
    items = load_scratch()
    if not items:
        return []

    current_turn = turn_count or datetime.datetime.now().timestamp()
    remaining = []

    for item in items:
        # Skip checked items
        if item.get('checked', False):
            continue

        # Check expiration
        created = datetime.datetime.fromisoformat(item['created'])
        age_turns = item.get('expires_at', NORMAL_EXPIRE_THRESHOLD)

        # Critical items get extended life
        criticality = item.get('criticality', 'normal')
        if criticality == 'critical':
            # Critical items persist longer but not forever
            if current_turn - created.timestamp() < CRITICAL_PRESERVE_THRESHOLD * 60:
                remaining.append(item)
        elif criticality == 'important':
            if current_turn - created.timestamp() < NORMAL_EXPIRE_THRESHOLD * 60:
                remaining.append(item)
        else:  # normal
            if current_turn - created.timestamp() < NORMAL_EXPIRE_THRESHOLD * 60:
                remaining.append(item)

    save_scratch(remaining)
    return remaining


# ─── L3: Semantic recall from scratch pad ───────────────────────

def scratch_recall(query: str, k: int = 3) -> List[dict]:
    """
    Query scratch pad via bud's semantic index for relevant items.

    Falls back gracefully to L2 JSON search if vector index unavailable.
    """
    try:
        from janus.memory import recall

        # Try semantic recall first (L3)
        results = recall(query, k=k)
        if results:
            return results
    except Exception:
        pass

    # Fallback: search L2 JSON directly
    items = load_scratch()
    results = []
    query_lower = query.lower()

    for item in items:
        content = item.get('content', '')
        if any(tag in content.lower() for tag in query_lower.split()):
            results.append({
                'text': content,
                'score': 0.5,  # Base confidence for fuzzy matches
                'metadata': item
            })

    # Sort by relevance (similarity to query)
    results.sort(key=lambda r: r.get('score', 0), reverse=True)
    return results[:k]


# ─── Skills Inventory Management ────────────────────────────────

# Condensed skills inventory (small description for system prompt)
SKILLS_INVENTORY = """
[SKILLS INVENTORY]
Process skills: brainstorming (idea generation), systematic-debugging (bug investigation), test-driven-development (TDD implementation), verification-before-completion (quality check), receiving-code-review (feedback handling), requesting-code-review (peer review), executing-plans (plan rollout), subagent-driven-development (parallel task execution), using-git-worktrees (isolation), writing-plans (multi-step planning), writing-skills (skill creation), using-superpowers (skill discovery).
Tool skills: frontend-design (UI/UX), huggingface-skills (HF APIs), firecrawl (web ops), pinecone (vector search), code-review (PR review), playground (interactive tools), claude-api (API apps).
[END SKILLS INVENTORY]
"""

def format_scratch_context() -> str:
    """
    Format active scratch items as a context string for system prompt.

    Returns empty string if no active items.
    """
    items = load_scratch()
    active = [i for i in items if not i.get('checked', False)]

    if not active:
        return ""

    lines = ["[SCRATCH PAD - ACTIVE ITEMS]"]
    for item in active:
        criticality = item.get('criticality', 'normal')
        marker = "!" if criticality == 'critical' else "?" if criticality == 'important' else "•"
        lines.append(f"  {marker} {item.get('content', '')}")

    lines.append("[END SCRATCH PAD]")
    return "\n".join(lines)


def format_skills_context() -> str:
    """
    Format skills inventory as a context string for system prompt.

    Returns a condensed overview of available skills.
    """
    return SKILLS_INVENTORY


# ─── Integration with memory compression ─────────────────────────

def maybe_compress_scratch(history: List[dict], scratch_pad: ScratchPad) -> List[dict]:
    """
    Compress scratch pad items when history exceeds threshold.

    Moves old scratch items to L2, keeps recent ones in L1.
    """
    if len(history) < NORMAL_EXPIRE_THRESHOLD:
        return history

    # Save active scratch to L2
    scratch_items = scratch_pad.to_dicts()
    if scratch_items:
        save_scratch(scratch_items)

    # Clear in-memory scratch
    scratch_pad.clear()

    return maybe_compress(history, summarize)


# ─── CLI Commands ───────────────────────────────────────────────

def note(content: str, tags: str = None, criticality: str = None) -> dict:
    """
    CLI command: Add a note to the scratch pad.

    Usage: !note <content> [--tag <tag>] [--critical]

    Args:
        content: The note content
        tags: Comma-separated tags
        criticality: 'critical', 'important', or None for normal

    Returns:
        The created ScratchItem as dict
    """
    scratch = ScratchPad()

    tag_list = tags.split(',') if tags else []
    crit = criticality if criticality in ('critical', 'important', 'normal') else 'normal'

    item = scratch.add(
        content=content,
        tags=tag_list,
        criticality=crit
    )

    # Persist to L2
    save_scratch([item.to_dict()])

    return item.to_dict()


def list_notes() -> List[dict]:
    """CLI command: List all scratch pad items."""
    return load_scratch()


def complete_note(item_id: str) -> bool:
    """CLI command: Mark a note as complete."""
    items = load_scratch()
    updated = False

    for item in items:
        if item.get('id') == item_id:
            item['checked'] = True
            updated = True
            break

    if updated:
        save_scratch(items)
    return updated


def clear_notes():
    """CLI command: Clear all scratch pad items."""
    save_scratch([])
