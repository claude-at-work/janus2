"""
JANUS Memory -- three-layer session persistence and recall.

L1: in-memory chat_history (live session, managed by shell.py)
L2: rolling JSON journal (~/.janus/context.json) -- last N turns, cross-session bridge
L3: bud semantic index -- long-term recall via embeddings (graceful fallback if not built)
L3+: symlink graph -- typed-edge traversal over pre-computed corpus links
     + semantic entry index (numpy cosine search → graph entry points)
     + connector MLP (joint quality classification)
     + flow MLP (path linearity classification)
     + connective tissue (voiced traversal phrases keyed by link type)
"""

import json
import math
import os
import random
import datetime
from pathlib import Path

import numpy as np

from janus.narrate_state import (
    NarrateState, TrajectorySignature,
    load_state, save_state, is_priming,
    blend_context_vec, blend_relevance,
    touch_or_push, fade_stack, consolidate,
)

# ─── Paths ────────────────────────────────────────────────────────
_JANUS_HOME = Path.home() / ".janus"
_INDEX_DIR = _JANUS_HOME / "index"
_WEIGHTS_DIR = _JANUS_HOME / "weights"
_LOGS_DIR = _JANUS_HOME / "logs"

_GRAPH_PATH = _INDEX_DIR / "symlink_index.json"
_ENTRY_EMB_PATH = _INDEX_DIR / "entry_embeddings.npy"
_ENTRY_IDS_PATH = _INDEX_DIR / "entry_chunk_ids.json"
_CONNECTOR_PATH = _WEIGHTS_DIR / "connector_mlp.json"
_FLOW_PATH = _WEIGHTS_DIR / "flow_mlp.json"
_TISSUE_PATH = _WEIGHTS_DIR / "connective_tissue.json"

# ─── Stopwords ────────────────────────────────────────────────────

_STOPWORDS = {
    "this", "that", "with", "have", "from", "they", "when", "will", "what",
    "your", "also", "just", "like", "more", "some", "than", "then", "them",
    "into", "only", "over", "very", "well", "much", "here", "time", "know",
    "been", "were", "there", "about", "would", "could", "should", "their",
    "which", "these", "those", "other", "after", "before", "through", "where",
    "while", "being", "having", "doing", "going", "getting", "making", "using",
}

# ─── Lazy-loaded singletons ──────────────────────────────────────

_graph_chunks: dict = {}       # chunk_id → {text, links, schema_tags}
_graph_word_index: dict = {}   # word → [chunk_id, ...]
_graph_loaded: bool = False

_entry_matrix: np.ndarray | None = None   # (N, 64) L2-normalised
_entry_ids: list[str] | None = None       # chunk_id per row
_entry_loaded: bool = False

_connector_mlp: dict | None = None
_flow_mlp: dict | None = None
_tissue: dict | None = None

_narrate_state: NarrateState | None = None    # lazy-loaded, persists across calls in session


# ─── Lightweight MLP forward pass ────────────────────────────────

def _mlp_forward(mlp: dict, x: list[float]) -> tuple[str, float]:
    """Run a trained MLP (JSON weights). Returns (label, confidence)."""
    current = np.array(x, dtype=np.float64)
    if len(current) < mlp["input_dim"]:
        current = np.pad(current, (0, mlp["input_dim"] - len(current)))
    elif len(current) > mlp["input_dim"]:
        current = current[:mlp["input_dim"]]

    for i, (W, b) in enumerate(zip(mlp["weights"], mlp["biases"])):
        W_arr = np.array(W, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        z = current @ W_arr + b_arr
        current = np.maximum(0.0, z) if i < len(mlp["weights"]) - 1 else z

    logits = current[:-1]
    shifted = logits - logits.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    conf = float(1.0 / (1.0 + np.exp(-np.clip(current[-1], -500, 500))))
    label = mlp["output_labels"][int(np.argmax(probs))]
    return label, conf


# ─── Loaders ─────────────────────────────────────────────────────

def _load_symlink_graph():
    global _graph_chunks, _graph_word_index, _graph_loaded
    _graph_loaded = True
    if not _GRAPH_PATH.exists():
        return
    try:
        with open(_GRAPH_PATH) as f:
            data = json.load(f)
        _graph_chunks = data.get("chunks", {})
        for chunk_id, chunk in _graph_chunks.items():
            words = chunk["text"].lower().split()
            for w in set(words[:120]):
                if len(w) > 3 and w not in _STOPWORDS:
                    _graph_word_index.setdefault(w, []).append(chunk_id)
    except Exception:
        _graph_chunks = {}
        _graph_word_index = {}


def _load_entry_index():
    global _entry_matrix, _entry_ids, _entry_loaded
    _entry_loaded = True
    if not _ENTRY_EMB_PATH.exists() or not _ENTRY_IDS_PATH.exists():
        return
    try:
        _entry_matrix = np.load(str(_ENTRY_EMB_PATH))
        with open(_ENTRY_IDS_PATH) as f:
            _entry_ids = json.load(f)
    except Exception:
        _entry_matrix = None
        _entry_ids = None


def _load_connector():
    global _connector_mlp
    if _CONNECTOR_PATH.exists():
        try:
            with open(_CONNECTOR_PATH) as f:
                _connector_mlp = json.load(f)
        except Exception:
            _connector_mlp = None


def _load_flow():
    global _flow_mlp
    if _FLOW_PATH.exists():
        try:
            with open(_FLOW_PATH) as f:
                _flow_mlp = json.load(f)
        except Exception:
            _flow_mlp = None


def _load_tissue():
    global _tissue
    if _TISSUE_PATH.exists():
        try:
            with open(_TISSUE_PATH) as f:
                _tissue = json.load(f)
        except Exception:
            _tissue = None


def _ensure_graph():
    if not _graph_loaded:
        _load_symlink_graph()
    if not _entry_loaded:
        _load_entry_index()


def _ensure_voice():
    if _connector_mlp is None:
        _load_connector()
    if _flow_mlp is None:
        _load_flow()
    if _tissue is None:
        _load_tissue()


# ─── Semantic entry (replaces word overlap when index exists) ────

def _reduce_dim(vec, target=64):
    if len(vec) <= target:
        return list(vec) + [0.0] * (target - len(vec))
    step = len(vec) / target
    return [float(vec[int(i * step)]) for i in range(target)]


def _embed_query_ollama(text: str) -> list[float] | None:
    """Embed via Ollama all-minilm:33m. Returns 64-dim reduced vector or None."""
    import urllib.request
    try:
        payload = json.dumps({"model": "all-minilm:33m", "input": text[:512]}).encode()
        req = urllib.request.Request(
            f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            embs = data.get("embeddings", [])
            if embs:
                return _reduce_dim(embs[0])
    except Exception:
        pass
    return None


def _semantic_entry(query: str, k: int = 3) -> list[str]:
    """
    Find graph entry points by cosine similarity against the entry index.
    Falls back to word overlap if semantic index isn't available.
    """
    _ensure_graph()

    if _entry_matrix is not None and _entry_ids is not None:
        qvec = _embed_query_ollama(query)
        if qvec is not None:
            q = np.array(qvec, dtype=np.float32).reshape(1, -1)
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
                scores = (q @ _entry_matrix.T).flatten()
                top_k = np.argsort(scores)[-k:][::-1]
                return [_entry_ids[i] for i in top_k if scores[i] > 0.15]

    # Fallback: word overlap
    query_words = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in _STOPWORDS]
    if not query_words:
        return []
    scores: dict[str, int] = {}
    for w in query_words:
        for cid in _graph_word_index.get(w, []):
            scores[cid] = scores.get(cid, 0) + 1
    return sorted(scores, key=lambda c: scores[c], reverse=True)[:k]


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


# ─── Voiced traversal ───────────────────────────────────────────

def _voice_joint(link_type: str, src_embedding: list | None, tgt_embedding: list | None) -> str:
    """
    Pick a transition phrase for a graph edge using:
    1. connector MLP (joint quality) for variety selection
    2. connective tissue (link type → Tyler's phrases)
    Falls back gracefully if MLPs or tissue aren't loaded.
    """
    _ensure_voice()

    phrases = [""]
    if _tissue:
        phrases = _tissue.get(link_type, _tissue.get("default", [""]))

    if _connector_mlp and src_embedding and tgt_embedding:
        # Connector MLP takes 256-dim (two 128-dim concatenated embeddings)
        # We have 64-dim embeddings, so pad each to 128
        padded_src = (src_embedding + [0.0] * 128)[:128]
        padded_tgt = (tgt_embedding + [0.0] * 128)[:128]
        try:
            connector_type, conf = _mlp_forward(_connector_mlp, padded_src + padded_tgt)
            # Use connector confidence to bias phrase selection:
            # high confidence → first phrase (most direct), low → later phrases (more tentative)
            idx = min(int((1.0 - conf) * len(phrases)), len(phrases) - 1)
            return phrases[idx]
        except Exception:
            pass

    return random.choice(phrases) if phrases else ""


def _get_chunk_embedding(chunk_id: str) -> list | None:
    """Get the stored embedding for a chunk from the entry index."""
    if _entry_matrix is None or _entry_ids is None:
        return None
    try:
        idx = _entry_ids.index(chunk_id)
        return _entry_matrix[idx].tolist()
    except (ValueError, IndexError):
        return None


# ─── Graph recall (upgraded) ─────────────────────────────────────

def _recall_from_graph(query: str, k: int = 3) -> list[dict]:
    """
    Find entry nodes by semantic similarity (with word-overlap fallback),
    then traverse one hop of outgoing typed links.
    Returns chunks with text, score, link_type.
    """
    _ensure_graph()
    if not _graph_chunks:
        return []

    top_entries = _semantic_entry(query, k=k)
    if not top_entries:
        return []

    results = []
    seen: set[str] = set()

    for entry_id in top_entries:
        if entry_id not in _graph_chunks or entry_id in seen:
            continue
        seen.add(entry_id)
        chunk = _graph_chunks[entry_id]
        results.append({
            "text": chunk["text"],
            "score": 0.8,
            "link_type": "entry",
        })

        links = sorted(chunk.get("links", []),
                       key=lambda l: l.get("confidence", 0), reverse=True)
        for link in links[:3]:
            if link.get("confidence", 0) < 0.35:
                break
            tgt_id = link.get("target")
            if tgt_id and tgt_id in _graph_chunks and tgt_id not in seen:
                seen.add(tgt_id)
                results.append({
                    "text": _graph_chunks[tgt_id]["text"],
                    "score": link.get("confidence", 0.5) * 0.85,
                    "link_type": link.get("type", ""),
                })

    return results[: k * 2]

JANUS_HOME = Path.home() / ".janus"
CONTEXT_FILE = JANUS_HOME / "context.json"
SESSIONS_DIR = JANUS_HOME / "sessions"
MAX_ROLLING_TURNS = 50       # turns kept in L2 rolling journal
MAX_CONTEXT_INJECT = 20      # turns injected into live history on startup
COMPRESSION_THRESHOLD = 40   # compress oldest half of history when live exceeds this


def _ensure_dirs():
    JANUS_HOME.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)


# ─── L2: Rolling JSON journal ─────────────────────────────────

def load_context() -> list[dict]:
    """Load the rolling journal and return the last MAX_CONTEXT_INJECT turns."""
    _ensure_dirs()
    try:
        with open(CONTEXT_FILE) as f:
            turns = json.load(f)
        if not isinstance(turns, list):
            return []
        return turns[-MAX_CONTEXT_INJECT:]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_context(history: list[dict]):
    """Persist the current history to the rolling journal, capped at MAX_ROLLING_TURNS."""
    _ensure_dirs()
    existing = []
    try:
        with open(CONTEXT_FILE) as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            existing = []
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    combined = existing + history
    trimmed = combined[-MAX_ROLLING_TURNS:]
    with open(CONTEXT_FILE, "w") as f:
        json.dump(trimmed, f, indent=2)


# ─── Session export (Claude-compatible format for bud) ────────

def export_session(history: list[dict], session_id: str | None = None):
    """
    Save the session in Claude conversation export format so bud can index it.
    Files land in ~/.janus/sessions/ and can be fed to `bud parse`.
    """
    if not history:
        return

    _ensure_dirs()
    sid = session_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"janus_{sid}.json"

    chat_messages = []
    for turn in history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            sender = "human"
        elif role == "assistant":
            sender = "assistant"
        else:
            continue
        chat_messages.append({
            "sender": sender,
            "content": [{"type": "text", "text": content}],
        })

    if not chat_messages:
        return

    export = [{
        "title": f"JANUS session {sid}",
        "created_at": datetime.datetime.now().isoformat(),
        "chat_messages": chat_messages,
    }]

    with open(path, "w") as f:
        json.dump(export, f, indent=2)

    return str(path)


# ─── L3: Bud semantic recall ──────────────────────────────────

def recall(query: str, k: int = 3) -> list[dict]:
    """
    Query bud's vector index for semantically relevant past context.
    Searches all *.faiss files in the index dir (skipping .bak.faiss).
    Returns a list of chunk dicts with 'text' and 'score' fields.
    Gracefully returns [] if the index hasn't been built yet.
    """
    try:
        import sys
        import glob
        _BUD_PATH = str(Path(__file__).parent.parent / "sandbox" / "root" / "scripts" / "bud-rag")
        if _BUD_PATH not in sys.path:
            sys.path.insert(0, _BUD_PATH)

        from bud.config import load_config
        from bud.lib.store import VectorStore
        from bud.lib.embeddings import EmbeddingClient

        config = load_config()
        output_dir = config.get("output_dir", str(JANUS_HOME / "bud_out"))
        index_dir = os.path.join(output_dir, "index")

        faiss_files = [
            f for f in glob.glob(os.path.join(index_dir, "*.faiss"))
            if not f.endswith(".bak.faiss")
        ]
        if not faiss_files:
            return []

        embed_cfg = config.get("embeddings", {})
        provider = embed_cfg.get("provider", "ollama")

        if provider == "voyage":
            import requests as _req
            api_key = embed_cfg.get("api_key", "")
            model = embed_cfg.get("model", "voyage-3")
            resp = _req.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"input": query, "model": model},
                timeout=30,
            )
            resp.raise_for_status()
            vector = resp.json()["data"][0]["embedding"]
        else:
            embedder = EmbeddingClient(config)
            vector = embedder.embed(query)

        seen_texts: set[str] = set()
        all_results: list[dict] = []
        for faiss_file in faiss_files:
            index_path = faiss_file[:-6]  # strip .faiss
            store = VectorStore(index_path)
            store.load()
            if store.count() == 0:
                continue
            for result in store.search(vector, k=k):
                text = result.get("text", result.get("content", ""))
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    all_results.append(result)

        all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return all_results[:k * 2]

    except Exception:
        return []


# ─── Wire 3: Slicer / Splicer conductor ──────────────────────────
# The query IS the conductor. Extractive intent → slice (raw chunks).
# Relational intent → splice (typed edge chain, topology visible).

_SPLICE_SIGNALS = {
    'relate', 'relationship', 'connect', 'between', 'compare',
    'how does', 'why does', 'explain', 'bridge', 'link', 'synthesize',
    'difference', 'similar', 'contrast', 'analogy', 'pattern',
}


def _splice_recall(query: str, k: int = 3) -> str | None:
    """
    Splice mode: build voiced relational chains from graph traversal.
    Shows the topology — not just what was said, but how ideas connect,
    narrated with connective tissue phrases and joint quality from MLPs.
    """
    _ensure_graph()
    if not _graph_chunks:
        return None

    top = _semantic_entry(query, k=k)
    if not top:
        return None

    # Classify path shape if flow MLP is available
    _ensure_voice()
    flow_label = None
    if _flow_mlp and _entry_matrix is not None and _entry_ids is not None:
        qvec = _embed_query_ollama(query)
        if qvec:
            # Flow MLP takes 152-dim: 64-dim query + 64-dim first entry + 24 metadata slots
            first_emb = _get_chunk_embedding(top[0]) if top else None
            if first_emb:
                flow_input = qvec + first_emb + [0.0] * 24
                try:
                    flow_label, _ = _mlp_forward(_flow_mlp, flow_input)
                except Exception:
                    pass

    chains = []
    seen: set[str] = set()

    for entry_id in top:
        if entry_id not in _graph_chunks or entry_id in seen:
            continue
        seen.add(entry_id)
        chunk = _graph_chunks[entry_id]
        node_text = chunk["text"][:80].replace('\n', ' ')
        src_emb = _get_chunk_embedding(entry_id)

        links = sorted(chunk.get("links", []),
                       key=lambda l: l.get("confidence", 0), reverse=True)
        if not links:
            chains.append(f'"{node_text}..."')
            continue

        link = links[0]
        tgt_id = link.get("target")
        link_type = link.get("type", "→")

        if tgt_id and tgt_id in _graph_chunks and tgt_id not in seen:
            seen.add(tgt_id)
            tgt_text = _graph_chunks[tgt_id]["text"][:80].replace('\n', ' ')
            tgt_emb = _get_chunk_embedding(tgt_id)
            voice = _voice_joint(link_type, src_emb, tgt_emb)

            # Check for onward link
            tgt_links = sorted(
                _graph_chunks[tgt_id].get("links", []),
                key=lambda l: l.get("confidence", 0), reverse=True
            )
            if tgt_links:
                tgt_link = tgt_links[0]
                tgt2_id = tgt_link.get("target")
                tgt2_type = tgt_link.get("type", "→")
                if tgt2_id and tgt2_id in _graph_chunks and tgt2_id not in seen:
                    seen.add(tgt2_id)
                    tgt2_text = _graph_chunks[tgt2_id]["text"][:60].replace('\n', ' ')
                    tgt2_emb = _get_chunk_embedding(tgt2_id)
                    voice2 = _voice_joint(tgt2_type, tgt_emb, tgt2_emb)
                    chains.append(
                        f'"{node_text}..."\n'
                        f'  {voice} [{link_type}]\n'
                        f'"{tgt_text}..."\n'
                        f'  {voice2} [{tgt2_type}]\n'
                        f'"{tgt2_text}..."'
                    )
                    continue

            chains.append(
                f'"{node_text}..."\n'
                f'  {voice} [{link_type}]\n'
                f'"{tgt_text}..."'
            )
        else:
            chains.append(f'"{node_text}..."')

    if not chains:
        return None

    header = "[Relational context from memory graph"
    if flow_label:
        header += f" — {flow_label} path"
    header += ":]"

    return "\n\n".join([header] + chains)


def conduct_recall(query: str, k: int = 3) -> str | None:
    """
    Query-driven conductor. Classifies intent, routes to slice or splice.
    Returns an ephemeral context string for injection into system_prompt.
    Gracefully returns None if no graph or no match.
    """
    lower = query.lower()
    is_splice = any(s in lower for s in _SPLICE_SIGNALS)
    return _splice_recall(query, k) if is_splice else recall_as_context(query, k)


def recall_as_context(query: str, k: int = 3) -> str | None:
    """
    Query the symlink graph (typed-edge traversal) and bud FAISS index,
    format results as a context string to inject into the system prompt.
    Graph results come first; FAISS fills in if graph is sparse.
    """
    graph_chunks = _recall_from_graph(query, k=k)
    faiss_chunks = recall(query, k=k)

    # Merge: graph first, then FAISS results not already covered
    graph_texts = {c["text"] for c in graph_chunks}
    combined = graph_chunks + [
        c for c in faiss_chunks
        if c.get("text", c.get("content", "")) not in graph_texts
    ]

    if not combined:
        return None

    lines = ["[Relevant context from past sessions:]"]
    for chunk in combined[:k + 2]:
        text = chunk.get("text", chunk.get("content", "")).strip()
        score = chunk.get("score", 0)
        link_type = chunk.get("link_type", "")
        min_score = 0.3
        if text and score > min_score:
            # Trim at sentence boundary instead of hard character chop
            trimmed = _extract_relevant(text, None, max_sentences=3)
            suffix = f" [{link_type}]" if link_type and link_type != "entry" else ""
            lines.append(f"- {trimmed}{suffix}")

    if len(lines) == 1:
        return None

    return "\n".join(lines)


# ─── Narrate: deep graph traversal as standalone response ────

PHI = (1.0 + math.sqrt(5.0)) / 2.0       # 1.6180339887...
PHI_INV = 1.0 / PHI                       # 0.6180339887...


def _cosine(a_vec: list | np.ndarray, b_vec: list | np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.array(a_vec, dtype=np.float32)
    b = np.array(b_vec, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _relevance_to_query(chunk_id: str, query_vec: list[float] | None) -> float:
    """Cosine similarity between a chunk and the query embedding."""
    if query_vec is None:
        return 0.5
    chunk_emb = _get_chunk_embedding(chunk_id)
    if chunk_emb is None:
        return 0.5
    return _cosine(query_vec, chunk_emb)


def _local_coherence(src_id: str, tgt_id: str) -> float:
    """Cosine similarity between two chunks — measures semantic continuity."""
    src_emb = _get_chunk_embedding(src_id)
    tgt_emb = _get_chunk_embedding(tgt_id)
    if src_emb is None or tgt_emb is None:
        return 0.5
    return _cosine(src_emb, tgt_emb)


# Expected semantic continuity per link type.
# High = should be semantically close. Low = discontinuity is natural.
_CONTINUITY_PROFILE = {
    'extension': 0.85,   # should flow directly
    'context':   0.75,   # grounding, close
    'resolution': 0.65,  # arriving somewhere — moderate continuity
    'analogy':   0.45,   # parallel structure, different territory
    'contrast':  0.30,   # tension is the point
    'metaphor':  0.20,   # distant jump, deep resonance
    'symbol':    0.25,   # recurring motif, may surface anywhere
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving bullet points and lists as units."""
    import re
    # First, collapse bullet/list blocks into single units
    # Replace bullet markers with a sentinel, split on sentence boundaries, then restore
    cleaned = text.strip()
    # Collapse markdown-style bullets and numbered lists into joined lines
    cleaned = re.sub(r'\n\s*[-•*]\s+', ' — ', cleaned)
    cleaned = re.sub(r'\n\s*\d+[.)]\s+', ' — ', cleaned)
    # Collapse single newlines (paragraph continuation) but preserve double newlines
    cleaned = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned)
    # Now split on sentence-ending punctuation or double newlines
    raw = re.split(r'(?<=[.!?])\s+|\n{2,}', cleaned)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 20]


def _extract_relevant(text: str, query_vec: list[float] | None,
                      max_sentences: int = 4) -> str:
    """
    Find the best *contiguous* window of sentences in a chunk.
    Keeps sentences in their natural order and context — no Frankenstein.
    Scores each window by signal density, picks the one that carries
    the most weight for the query.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text[:400].strip()

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score each sentence
    scores = []
    for i, s in enumerate(sentences):
        score = 0.0
        slen = len(s)
        if 40 < slen < 300:
            score += 0.3
        if not s.rstrip().endswith('?'):
            score += 0.2
        if any(w in s.lower() for w in ['because', 'means', 'realize', 'the point',
                                         'actually', 'what if', 'essentially',
                                         'the thing', 'notice', 'recognize']):
            score += 0.15
        scores.append(score)

    # Slide a contiguous window and pick the one with the highest total score
    best_start = 0
    best_score = 0.0
    # Normalise heuristic scores so they blend fairly with cosine similarity
    max_heuristic = max(
        (sum(scores[s:s + max_sentences]) + max(0, 0.2 - s * 0.03))
        for s in range(len(sentences) - max_sentences + 1)
    ) or 1.0
    for start in range(len(sentences) - max_sentences + 1):
        heuristic = sum(scores[start:start + max_sentences])
        # Small bonus for starting near the top (thesis tends to be early)
        heuristic += max(0, 0.2 - start * 0.03)
        heuristic_norm = heuristic / max_heuristic
        if query_vec is not None:
            window_text = " ".join(sentences[start:start + max_sentences])
            w_emb = _embed_query_ollama(window_text)
            q_sim = _cosine(w_emb, query_vec) if w_emb else 0.0
            window_score = 0.6 * heuristic_norm + 0.4 * q_sim
        else:
            window_score = heuristic_norm
        if window_score > best_score:
            best_score = window_score
            best_start = start

    return " ".join(sentences[best_start:best_start + max_sentences])


def _find_bridge(src_id: str, tgt_id: str, link_type: str,
                  seen: set[str]) -> str | None:
    """
    When the semantic gap between src and tgt is too wide for the link type,
    find a bridge chunk — a structurally similar node that sits between them
    semantically. The bridge acts as connective tissue made of real corpus text.
    """
    src_emb = _get_chunk_embedding(src_id)
    tgt_emb = _get_chunk_embedding(tgt_id)
    if src_emb is None or tgt_emb is None:
        return None

    # The bridge should be close to the midpoint between src and tgt
    midpoint = ((np.array(src_emb) + np.array(tgt_emb)) / 2.0).tolist()

    # Search the entry index for chunks near the midpoint
    if _entry_matrix is None or _entry_ids is None:
        return None

    q = np.array(midpoint, dtype=np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm == 0:
        return None
    q = q / norm
    scores = (q @ _entry_matrix.T).flatten()
    top_k = np.argsort(scores)[-10:][::-1]

    for idx in top_k:
        cid = _entry_ids[idx]
        if cid in seen or cid == src_id or cid == tgt_id:
            continue
        if cid not in _graph_chunks:
            continue
        # Bridge should be closer to both src and tgt than they are to each other
        bridge_to_src = _cosine(_get_chunk_embedding(cid), src_emb)
        bridge_to_tgt = _cosine(_get_chunk_embedding(cid), tgt_emb)
        if bridge_to_src > 0.2 and bridge_to_tgt > 0.2:
            return cid

    return None


def narrate(query: str, depth: int = 5, branches: int = 2,
            relevance_floor: float = 0.15,
            entry_threshold: float = 0.25) -> str | None:
    """
    Walk the graph and let it speak. No LLM synthesis — the traversal IS the response.

    Three-signal traversal:
      1. Edge confidence (structural topology)
      2. Query relevance (global direction)
      3. Local coherence (semantic continuity with current position)

    When the semantic gap between hops is wider than the link type expects,
    a bridge chunk is inserted — a structurally similar node from the corpus
    that acts as real connective tissue. When semantics are tight, inertia
    carries and the tissue whispers. Phi governs the unfolding.

    Returns None with grace if the graph doesn't hold the queried territory.
    """
    _ensure_graph()
    if not _graph_chunks:
        return None

    # ── Load context state (once per session, reuse in memory) ──
    global _narrate_state
    if _narrate_state is None:
        _narrate_state = load_state()
    _state = _narrate_state
    priming = is_priming(query)

    # In priming mode with prior context, use context vec for entry search
    if priming and _state.session_context is not None:
        entries = _semantic_entry_by_vec(_state.session_context, k=branches)
    else:
        entries = _semantic_entry(query, k=branches)
    if not entries:
        return None

    _ensure_voice()
    query_vec = _embed_query_ollama(query) if not priming else None

    # Prefer human-side entries — enter through Tyler's pattern, cross to AI response.
    # Re-sort entries so human chunks come first, but don't discard assistant entries.
    def _entry_sort_key(eid):
        sender = _graph_chunks.get(eid, {}).get("schema_tags", {}).get("sender", "")
        return 0 if sender == "human" else 1
    entries = sorted(entries, key=_entry_sort_key)

    seen: set[str] = set()
    threads: list[str] = []
    all_paths: list[list[dict]] = []  # raw path data for logging
    context_vec = _state.session_context  # outer scope for state save
    trajectory = TrajectorySignature()

    for entry_id in entries:
        if entry_id not in _graph_chunks or entry_id in seen:
            continue

        path = []
        current_id = entry_id
        momentum = 1.0  # tracks inertia — high = semantically smooth run
        context_vec = _state.session_context  # warm start from prior walk (or None)
        trajectory = TrajectorySignature()

        for hop in range(depth):
            if current_id in seen or current_id not in _graph_chunks:
                break

            # Check relevance to query — stop if we've drifted
            if hop > 0:
                if priming:
                    if context_vec is not None:
                        chunk_emb_check = _get_chunk_embedding(current_id)
                        rel = _cosine(chunk_emb_check, context_vec) if chunk_emb_check else 0.5
                    else:
                        rel = 0.5  # no signal yet, keep walking
                else:
                    chunk_emb_check = _get_chunk_embedding(current_id)
                    if chunk_emb_check is not None:
                        rel = blend_relevance(chunk_emb_check, query_vec, context_vec, priming=False)
                    else:
                        rel = _relevance_to_query(current_id, query_vec)
                if not priming and rel < relevance_floor:
                    break

            seen.add(current_id)
            chunk = _graph_chunks[current_id]
            text = _extract_relevant(chunk["text"].strip(), query_vec)
            current_sender = chunk.get("schema_tags", {}).get("sender", "")

            path.append({"id": current_id, "text": text,
                         "link_type": None, "voice": None, "is_bridge": False,
                         "sender": current_sender})

            # ── Update running context ──
            chunk_emb = _get_chunk_embedding(current_id)
            if chunk_emb is not None:
                context_vec = blend_context_vec(context_vec, chunk_emb)

            # ── Score candidates with four signals ──
            # The fourth signal: speaker rhythm.
            # Conversations breathe: human → assistant → human → assistant.
            # The traversal should prefer crossing the membrane at the right time.
            links = chunk.get("links", [])
            candidates = []

            # Count recent same-speaker hops to know if we're due for a crossing
            recent_senders = [p.get("sender", "") for p in path[-3:] if p.get("sender")]
            same_speaker_run = 0
            for s in reversed(recent_senders):
                if s == current_sender:
                    same_speaker_run += 1
                else:
                    break

            for l in links:
                tgt = l.get("target")
                if not (tgt and tgt in _graph_chunks
                        and tgt not in seen
                        and l.get("confidence", 0) >= 0.3):
                    continue

                link_type = l.get("type", "extension")
                conf = l.get("confidence", 0.5)
                tgt_emb = _get_chunk_embedding(tgt)
                if tgt_emb is not None:
                    q_rel = blend_relevance(tgt_emb, query_vec, context_vec, priming=priming)
                else:
                    q_rel = _relevance_to_query(tgt, query_vec) if not priming else 0.5
                coherence = _local_coherence(current_id, tgt)

                # Speaker rhythm: bonus for crossing after same-speaker runs
                tgt_sender = _graph_chunks[tgt].get("schema_tags", {}).get("sender", "")
                crossing = tgt_sender != current_sender and tgt_sender != ""
                # The longer we've been in one voice, the stronger the pull to cross
                cross_bonus = 0.0
                if crossing and same_speaker_run >= 1:
                    cross_bonus = min(0.25, same_speaker_run * 0.1)
                # Slight penalty for staying in same voice too long (>2 hops)
                elif not crossing and same_speaker_run >= 2:
                    cross_bonus = -0.08

                # Trajectory signature scoring
                traj_bonus = trajectory.score_candidate(link_type, crossing)

                # Stack impression bias — faint pull toward remembered territory
                stack_bias = 0.0
                if tgt_emb is not None:
                    for imp in _state.stack:
                        imp_sim = _cosine(tgt_emb, imp.vector)
                        stack_bias += imp_sim * imp.strength * 0.03  # faint
                    if _state.residue is not None:
                        res_sim = _cosine(tgt_emb, _state.residue)
                        stack_bias += res_sim * 0.02  # even fainter

                # Weight blend shifts with depth — deeper = more semantic, less structural
                d = hop / max(depth - 1, 1)
                structural_w = PHI_INV * (1 - d)    # starts ~0.618, fades
                query_w = 0.3                        # constant pull toward intent
                coherence_w = PHI_INV * d            # starts 0, grows to ~0.618

                score = (conf * structural_w
                         + q_rel * query_w
                         + coherence * coherence_w
                         + cross_bonus
                         + traj_bonus
                         + stack_bias)

                candidates.append((l, score, coherence))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1], reverse=True)

            # Pick best candidate, preferring link type diversity
            used_types = {p["link_type"] for p in path if p["link_type"]}
            chosen_link = candidates[0][0]
            chosen_coherence = candidates[0][2]
            for c, score, coh in candidates[:5]:
                if c.get("type") not in used_types:
                    chosen_link = c
                    chosen_coherence = coh
                    break

            link_type = chosen_link.get("type", "extension")
            tgt_id = chosen_link["target"]

            # Record trajectory step
            tgt_sender_for_sig = _graph_chunks[tgt_id].get("schema_tags", {}).get("sender", "")
            sig_crossing = tgt_sender_for_sig != current_sender and tgt_sender_for_sig != ""
            trajectory.record(link_type, sig_crossing)

            # ── Semantic gap check ──
            expected_continuity = _CONTINUITY_PROFILE.get(link_type, 0.5)
            gap = expected_continuity - chosen_coherence

            if gap > 0.25:
                # Wide semantic gap — find a bridge chunk
                bridge_id = _find_bridge(current_id, tgt_id, link_type, seen)
                if bridge_id:
                    seen.add(bridge_id)
                    bridge_chunk = _graph_chunks[bridge_id]
                    bridge_text = _extract_relevant(bridge_chunk["text"].strip(), query_vec,
                                                    max_sentences=2)
                    # Voice into the bridge
                    src_emb = _get_chunk_embedding(current_id)
                    bridge_emb = _get_chunk_embedding(bridge_id)
                    bridge_voice = _voice_joint(link_type, src_emb, bridge_emb)

                    path[-1]["link_type"] = link_type
                    path[-1]["voice"] = bridge_voice

                    path.append({"id": bridge_id, "text": bridge_text,
                                 "link_type": link_type, "voice": None,
                                 "is_bridge": True})
                    momentum *= PHI_INV  # gap costs momentum
                else:
                    # No bridge found — use tissue phrase, accept the gap
                    src_emb = _get_chunk_embedding(current_id)
                    tgt_emb = _get_chunk_embedding(tgt_id)
                    voice = _voice_joint(link_type, src_emb, tgt_emb)
                    path[-1]["link_type"] = link_type
                    path[-1]["voice"] = voice
                    momentum *= PHI_INV
            elif gap > 0.1:
                # Moderate gap — tissue phrase bridges it
                src_emb = _get_chunk_embedding(current_id)
                tgt_emb = _get_chunk_embedding(tgt_id)
                voice = _voice_joint(link_type, src_emb, tgt_emb)
                path[-1]["link_type"] = link_type
                path[-1]["voice"] = voice
                momentum = max(momentum * 0.9, 0.3)
            else:
                # Tight semantics — inertia carries, tissue whispers or stays silent
                if momentum > PHI_INV:
                    # High momentum: no tissue, just flow
                    path[-1]["link_type"] = link_type
                    path[-1]["voice"] = None
                else:
                    # Low momentum: gentle tissue to re-establish rhythm
                    src_emb = _get_chunk_embedding(current_id)
                    tgt_emb = _get_chunk_embedding(tgt_id)
                    voice = _voice_joint(link_type, src_emb, tgt_emb)
                    path[-1]["link_type"] = link_type
                    path[-1]["voice"] = voice
                momentum = min(momentum * PHI, 1.0)  # tight hop builds momentum toward phi

            current_id = tgt_id

        if not path:
            continue

        # ── Path coherence check ──
        # Short queries (poetic, minimal) can produce short but potent paths.
        # Longer queries should produce longer paths to be worth showing.
        content_hops = [p for p in path if not p.get("is_bridge")]
        min_hops = 1 if len(query.strip()) < 10 else 2
        if len(content_hops) < min_hops:
            continue
        # Check the path actually went somewhere — measure semantic distance
        # between first and last hop. If they're too similar, we orbited in place.
        if len(content_hops) >= 3:
            first_emb = _get_chunk_embedding(content_hops[0]["id"])
            last_emb = _get_chunk_embedding(content_hops[-1]["id"])
            if first_emb and last_emb:
                traversal_distance = 1.0 - _cosine(first_emb, last_emb)
                if traversal_distance < 0.05:
                    continue  # went nowhere — orbited in place

        all_paths.append(list(path))

        # ── Render thread ──
        parts = []
        for i, node in enumerate(path):
            parts.append(node["text"])
            if node["voice"] and i < len(path) - 1:
                parts.append(f"\n  *{node['voice']}*\n")
            elif i < len(path) - 1 and not node["voice"]:
                parts.append("")  # tight flow, just a breath

        threads.append("\n".join(parts))

    if not threads:
        return None

    # Minimum viable response — if the total content is too thin,
    # the graph touched the territory but didn't have enough to say.
    total_content = sum(len(t) for t in threads)
    content_lines = sum(
        1 for t in threads
        for line in t.split('\n')
        if line.strip() and not line.strip().startswith('*')
    )
    if total_content < 150 or content_lines < 2:
        return None

    if len(threads) == 1:
        result = threads[0]
    else:
        output = []
        for i, thread in enumerate(threads):
            if i > 0:
                output.append("\n─ · ─\n")
            output.append(thread)
        result = "\n".join(output)

    # ── Log to ~/.janus/logs/narrate_history.jsonl ──
    try:
        from datetime import datetime, timezone
        log_path = str(_LOGS_DIR / "narrate_history.jsonl")
        traversal_log = []
        for thread_paths in all_paths:
            hops = []
            for node in thread_paths:
                chunk = _graph_chunks.get(node["id"], {})
                tags = chunk.get("schema_tags", {})
                hops.append({
                    "chunk_id": node["id"],
                    "sender": node.get("sender", tags.get("sender", "")),
                    "conversation": tags.get("conv", ""),
                    "turn_id": tags.get("turn_id", ""),
                    "link_type": node.get("link_type"),
                    "voice": node.get("voice"),
                    "is_bridge": node.get("is_bridge", False),
                })
            traversal_log.append(hops)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "depth": depth,
            "branches": branches,
            "threads": len(threads),
            "total_hops": sum(len(t) for t in all_paths),
            "traversal": traversal_log,
            "response": result,
        }
        entry["context_influence"] = {
            "mode": "priming" if priming else "query",
            "context_vec_norm": float(np.linalg.norm(context_vec)) if context_vec is not None else 0.0,
            "stack_size": len(_state.stack),
            "residue_present": _state.residue is not None,
        }
        entry["trajectory_signature"] = trajectory.to_list() if trajectory else []
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging should never break narrate

    # ── Update context state ──
    if context_vec is not None:
        _state.session_context = context_vec
        walk_coherence = min(1.0, sum(len(p) for p in all_paths) / (depth * len(all_paths))) if all_paths else 0.5
        touch_or_push(_state, context_vec, trajectory, walk_coherence=walk_coherence)
        fade_stack(_state, residue_vec=_state.residue)
        consolidate(_state)
        save_state(_state)

    return result


# ─── Narrate-for-context: lean graph walk for LLM pre-context ─

def narrate_context(query: str, max_chars: int = 1200) -> str | None:
    """
    Shallow, compact narrate pass that returns a context block for LLM injection.

    Depth 3, single branch, no bridge chunks surfaced, voice phrases stripped.
    The result is a series of chunk excerpts with link-type labels — enough for
    the LLM to draw on without flooding the context window.

    Returns None if the graph has nothing relevant.
    """
    _ensure_graph()
    if not _graph_chunks:
        return None

    global _narrate_state
    if _narrate_state is None:
        _narrate_state = load_state()
    _state = _narrate_state
    priming = is_priming(query)

    if priming and _state.session_context is not None:
        entries = _semantic_entry_by_vec(_state.session_context, k=1)
    else:
        entries = _semantic_entry(query, k=1)
    if not entries:
        return None

    query_vec = _embed_query_ollama(query) if not priming else None
    context_vec = _state.session_context
    seen: set[str] = set()
    excerpts: list[str] = []
    total_len = 0

    for entry_id in entries[:1]:  # single branch
        if entry_id not in _graph_chunks:
            continue

        current_id = entry_id
        for hop in range(3):
            if current_id in seen or current_id not in _graph_chunks:
                break
            if hop > 0:
                chunk_emb_check = _get_chunk_embedding(current_id)
                if chunk_emb_check is not None:
                    rel = blend_relevance(chunk_emb_check, query_vec, context_vec, priming=priming)
                else:
                    rel = _relevance_to_query(current_id, query_vec) if not priming else 0.5
                if not priming and rel < 0.15:
                    break

            seen.add(current_id)
            chunk = _graph_chunks[current_id]
            text = _extract_relevant(chunk["text"].strip(), query_vec, max_sentences=2)

            if not text or len(text) < 20:
                break

            # Find the link type leading to next hop (for labeling)
            link_label = ""
            links = chunk.get("links", [])
            best_link = None
            best_score = -1.0
            for l in links:
                tgt = l.get("target")
                if tgt and tgt in _graph_chunks and tgt not in seen:
                    conf = l.get("confidence", 0)
                    tgt_emb = _get_chunk_embedding(tgt)
                    if tgt_emb is not None:
                        q_rel = blend_relevance(tgt_emb, query_vec, context_vec, priming=priming)
                    else:
                        q_rel = _relevance_to_query(tgt, query_vec) if not priming else 0.5
                    s = conf * 0.5 + q_rel * 0.5
                    if s > best_score:
                        best_score = s
                        best_link = l

            if hop > 0:
                link_label = f" [{best_link.get('type', 'extension') if best_link else 'extension'}]"

            excerpt = f"- {text}{link_label}"
            if total_len + len(excerpt) > max_chars:
                break
            excerpts.append(excerpt)
            total_len += len(excerpt)

            if best_link:
                current_id = best_link["target"]
            else:
                break

    if not excerpts:
        return None

    header = "[Graph memory — use what's relevant, discard what isn't:]"
    return header + "\n" + "\n".join(excerpts)


# ─── In-session compression ───────────────────────────────────

def maybe_compress(history: list[dict], summarize_fn) -> list[dict]:
    """
    If history exceeds COMPRESSION_THRESHOLD, summarize the oldest half
    into a single context entry and keep the recent half intact.
    summarize_fn: callable(text: str) -> str  (uses the LLM)
    """
    if len(history) < COMPRESSION_THRESHOLD:
        return history

    mid = len(history) // 2
    old_turns = history[:mid]
    recent_turns = history[mid:]

    old_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in old_turns
    )

    try:
        summary = summarize_fn(old_text)
    except Exception:
        summary = f"[Earlier conversation: {len(old_turns)} turns]"

    compressed = [{"role": "system", "content": f"[Context summary from earlier in this session]:\n{summary}"}]
    return compressed + recent_turns
