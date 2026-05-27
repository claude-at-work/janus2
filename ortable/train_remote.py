#!/usr/bin/env python3
"""
JANUS Link MLP — Portable Training Script
==========================================

Self-contained. No JANUS imports. Ships to any machine with numpy.

Embeds the full conversation corpus, generates grounded training pairs,
trains a 128-dim link-prediction MLP with phi-resonance aesthetic loss
and convergence-aware pair compression.

Embedding backend is swappable — see EMBED_BACKEND below.

Setup & Run:
    See INSTRUCTIONS.md in this directory.
"""
import json
import sys
import math
import time
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# ─── Config ───────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "embedding_cache.jsonl"
CHECKPOINT_PATH = DATA_DIR / "embed_checkpoint.json"
WEIGHTS_PATH = Path(__file__).parent / "link_mlp_trained.json"
BACKUP_PATH = Path(__file__).parent / "link_mlp_trained.backup.json"
PAIRS_PATH = DATA_DIR / "training_pairs.json"

# The irrational constants — inlined from trainer.py
PHI = (1.0 + math.sqrt(5.0)) / 2.0       # 1.6180339887...
PHI_INV = 1.0 / PHI                        # 0.6180339887...

LINK_TYPES = ['analogy', 'contrast', 'extension', 'context',
              'resolution', 'metaphor', 'symbol']
TARGET_DIM = 64   # per-chunk after reduce_dim; pairs = 128
TOTAL_CORPUS = 6349  # known corpus size for projection stats


# ─── Embedding Backend ────────────────────────────────────────────
#
# Set EMBED_BACKEND to one of:
#   "sentence_transformers"  — local GPU, fastest (default)
#   "ollama"                 — local Ollama instance
#   "voyage"                 — Voyage AI API
#
# Each backend implements embed_text(str) -> List[float]

EMBED_BACKEND = "sentence_transformers"

# Voyage AI config (only if EMBED_BACKEND == "voyage")
VOYAGE_API_KEY = ""           # set this or use env VOYAGE_API_KEY
VOYAGE_MODEL = "voyage-3"  # 1024-dim, matches buddy-cli embedding space

# Ollama config (only if EMBED_BACKEND == "ollama")
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "all-minilm:33m"

# sentence-transformers config
ST_MODEL = "all-MiniLM-L6-v2"  # 384-dim, same model as ollama all-minilm:33m

# ─── lazy-loaded embedding singleton ─────────────────────────────
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(ST_MODEL)
        if hasattr(_st_model, 'to'):
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _st_model = _st_model.to(device)
            print(f"  sentence-transformers on {device}")
    return _st_model


def embed_text(text: str) -> List[float]:
    """Embed a single text. Backend-agnostic."""
    if EMBED_BACKEND == "sentence_transformers":
        model = _get_st_model()
        vec = model.encode(text, show_progress_bar=False)
        return vec.tolist() if hasattr(vec, 'tolist') else list(vec)

    elif EMBED_BACKEND == "ollama":
        import urllib.request
        payload = json.dumps({"model": OLLAMA_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            embs = data.get("embeddings", [])
            return embs[0] if embs else []

    elif EMBED_BACKEND == "voyage":
        import os, urllib.request
        api_key = VOYAGE_API_KEY or os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            raise ValueError("Set VOYAGE_API_KEY or env var VOYAGE_API_KEY")
        payload = json.dumps({
            "model": VOYAGE_MODEL,
            "input": [text],
        }).encode()
        req = urllib.request.Request(
            "https://api.voyageai.com/v1/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data["data"][0]["embedding"]

    else:
        raise ValueError(f"Unknown EMBED_BACKEND: {EMBED_BACKEND}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Batch embed for backends that support it.
    Falls back to sequential for those that don't.
    """
    if EMBED_BACKEND == "sentence_transformers":
        model = _get_st_model()
        vecs = model.encode(texts, show_progress_bar=False, batch_size=64)
        return [v.tolist() if hasattr(v, 'tolist') else list(v) for v in vecs]

    elif EMBED_BACKEND == "voyage":
        import os, urllib.request
        api_key = VOYAGE_API_KEY or os.environ.get("VOYAGE_API_KEY", "")
        if not api_key:
            raise ValueError("Set VOYAGE_API_KEY or env var VOYAGE_API_KEY")
        # Voyage supports batch up to 128
        results = []
        for i in range(0, len(texts), 128):
            batch = texts[i:i+128]
            payload = json.dumps({
                "model": VOYAGE_MODEL,
                "input": batch,
            }).encode()
            req = urllib.request.Request(
                "https://api.voyageai.com/v1/embeddings",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                results.extend([d["embedding"] for d in data["data"]])
        return results

    else:
        # Sequential fallback
        return [embed_text(t) for t in texts]


def l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector in place (matches FAISS storage)."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-10:
        return vec
    return [v / norm for v in vec]


def reduce_dim(vec: List[float], target: int = TARGET_DIM) -> List[float]:
    """L2-normalize then stride-sample down to target dimensions."""
    vec = l2_normalize(vec)
    if len(vec) <= target:
        return vec + [0.0] * (target - len(vec))
    step = len(vec) / target
    return [vec[int(i * step)] for i in range(target)]


def extract_top_of_mind(memory_context: str, max_chars: int = 180) -> str:
    """Pull the 'Top of mind' paragraph from memory_context markdown."""
    if not memory_context:
        return ""
    marker = "**Top of mind**"
    idx = memory_context.find(marker)
    if idx == -1:
        return ""
    rest = memory_context[idx + len(marker):].lstrip('\n ')
    # Stop at next section header
    next_header = rest.find("**")
    if next_header != -1:
        rest = rest[:next_header]
    return rest.strip()[:max_chars]


# ─── Embedding Cache ──────────────────────────────────────────────

class EmbeddingCache:
    """Persistent hash-keyed embedding cache. Append-only JSONL."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self._cache: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    self._cache[rec['k']] = rec['v']
                except (json.JSONDecodeError, KeyError):
                    continue

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(
            text[:512].encode('utf-8', errors='replace')
        ).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(self._key(text))

    def put(self, text: str, embedding: List[float]):
        k = self._key(text)
        if k in self._cache:
            return
        self._cache[k] = embedding
        with open(self.path, 'a') as f:
            f.write(json.dumps({'k': k, 'v': embedding}) + '\n')

    def put_batch(self, texts: List[str], embeddings: List[List[float]]):
        """Batch insert, one flush."""
        lines = []
        for text, emb in zip(texts, embeddings):
            k = self._key(text)
            if k not in self._cache:
                self._cache[k] = emb
                lines.append(json.dumps({'k': k, 'v': emb}))
        if lines:
            with open(self.path, 'a') as f:
                f.write('\n'.join(lines) + '\n')

    def __len__(self):
        return len(self._cache)


# ─── Checkpoint ───────────────────────────────────────────────────

class EmbedCheckpoint:
    """Tracks embedding frontier."""

    def __init__(self, path: Path = CHECKPOINT_PATH):
        self.path = path
        self.total_embedded = 0
        self.total_turns_seen = 0
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.total_embedded = d.get('total_embedded', 0)
            self.total_turns_seen = d.get('total_turns_seen', 0)
        except (json.JSONDecodeError, KeyError):
            pass

    def save(self, embedded: int, seen: int):
        self.total_embedded = embedded
        self.total_turns_seen = seen
        with open(self.path, 'w') as f:
            json.dump({
                'total_embedded': self.total_embedded,
                'total_turns_seen': self.total_turns_seen,
                'timestamp': time.time(),
            }, f)

    def reset(self):
        self.total_embedded = 0
        self.total_turns_seen = 0
        if self.path.exists():
            self.path.unlink()


# ─── Corpus Loading + Cached Embedding ────────────────────────────

def load_and_embed_turns(max_turns: int, cache: EmbeddingCache,
                         ckpt: EmbedCheckpoint, batch_size: int = 64
                         ) -> List[Dict]:
    """
    Walk the full corpus. Cache hits are instant. New texts are
    batched for GPU-friendly throughput.
    """
    files = sorted(DATA_DIR.glob("conversations_*.jsonl"))
    turns = []
    pending_texts = []
    pending_indices = []
    cache_hits = 0
    new_embeds = 0

    if ckpt.total_embedded > 0:
        print(f"  prior frontier: {ckpt.total_embedded} cached, "
              f"{ckpt.total_turns_seen} seen")
        sys.stdout.flush()

    for jf in files:
        with open(jf) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                conv_name = rec.get("conversation_name", "").strip()
                top_of_mind = extract_top_of_mind(rec.get("memory_context", ""))
                for t in rec.get("turns", []):
                    text = t.get("text", "").strip()
                    if not text or len(text) <= 20:
                        continue

                    text_trunc = text[:512]
                    # Ground the embedding in conversation context:
                    #   [conv_name] [top of mind] turn_text
                    # Named conversations and active cognitive state both shift
                    # where the turn lands in embedding space.
                    parts = []
                    if conv_name:
                        parts.append(f"[{conv_name}]")
                    if top_of_mind:
                        parts.append(f"[{top_of_mind}]")
                    parts.append(text_trunc)
                    embed_text_key = " ".join(parts)
                    cached = cache.get(embed_text_key)

                    turn_entry = {
                        "text": text_trunc,
                        "sender": t.get("sender", "?"),
                        "conv": conv_name,
                        "tags": {},
                        "link_to_next": None,
                        "embedding": None,
                    }

                    if cached is not None:
                        turn_entry["embedding"] = cached
                        cache_hits += 1
                    else:
                        # Queue for batch embedding
                        pending_texts.append(embed_text_key)
                        pending_indices.append(len(turns))

                    turns.append(turn_entry)
                    n = len(turns)

                    # Flush batch when full
                    if len(pending_texts) >= batch_size:
                        vecs = embed_batch(pending_texts)
                        reduced = [reduce_dim(v) for v in vecs]
                        cache.put_batch(pending_texts, reduced)
                        for idx, emb in zip(pending_indices, reduced):
                            turns[idx]["embedding"] = emb
                        new_embeds += len(pending_texts)
                        pending_texts.clear()
                        pending_indices.clear()

                    if n % 100 == 0 or n >= max_turns:
                        sys.stdout.write(
                            f"\r  turns: {n}  cached: {cache_hits}  "
                            f"frontier: {new_embeds}  "
                            f"cache: {len(cache)}"
                        )
                        sys.stdout.flush()
                        ckpt.save(embedded=len(cache), seen=n)

                    if n >= max_turns:
                        break
                if len(turns) >= max_turns:
                    break
        if len(turns) >= max_turns:
            break

    # Flush remaining
    if pending_texts:
        vecs = embed_batch(pending_texts)
        reduced = [reduce_dim(v) for v in vecs]
        cache.put_batch(pending_texts, reduced)
        for idx, emb in zip(pending_indices, reduced):
            turns[idx]["embedding"] = emb
        new_embeds += len(pending_texts)

    print()
    ckpt.save(embedded=len(cache), seen=len(turns))
    return turns


# ─── Cosine Similarity ───────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ─── Positional Links (grounded in observables) ──────────────────

def add_positional_links(turns: List[Dict]) -> List[Dict]:
    """
    Label adjacent pairs from observable features:
    speaker transition x cosine similarity -> link type.
    """
    for i in range(len(turns) - 1):
        t1, t2 = turns[i], turns[i + 1]
        if t1["conv"] != t2["conv"]:
            continue

        sim = _cosine_sim(t1["embedding"], t2["embedding"])
        same_sender = t1["sender"] == t2["sender"]

        if same_sender and sim > 0.4:
            link_type, strength = "extension", 0.5 + sim * 0.4
        elif same_sender:
            link_type, strength = "context", 0.3 + sim * 0.3
        elif sim > 0.5:
            link_type, strength = "resolution", 0.5 + sim * 0.3
        else:
            link_type, strength = "contrast", 0.4 + (1.0 - sim) * 0.3

        turns[i]["link_to_next"] = {
            "type": link_type,
            "strength": min(1.0, strength),
        }
    return turns


# ─── Training Pair Generation ────────────────────────────────────

def generate_training_pairs(turns: List[Dict],
                            max_pairs: int = 5000) -> List[Dict]:
    pairs = []
    indexed = [t for t in turns if t.get('embedding')]

    if len(indexed) < 2:
        return []

    def _depth_confidences(i, j, indexed, hops=3):
        """Confidences at increasing distances from anchor i toward j."""
        n = len(indexed)
        direction = 1 if j > i else -1
        dc = []
        for h in range(1, hops + 1):
            k = i + direction * h
            if 0 <= k < n:
                sim = _cosine_sim(indexed[i]['embedding'], indexed[k]['embedding'])
                dc.append(max(0.05, sim))
        return dc if len(dc) >= 2 else None

    # Positional pairs
    for i, t in enumerate(indexed):
        link = t.get('link_to_next')
        if link and i + 1 < len(indexed):
            t2 = indexed[i + 1]
            pairs.append({
                'input': t['embedding'] + t2['embedding'],
                'label': link['type'],
                'confidence': link['strength'],
                'source': 'positional',
                'depth_confidences': _depth_confidences(i, i + 1, indexed),
            })

    # Non-adjacent discovered pairs
    n = len(indexed)
    target = min(max_pairs - len(pairs), n * (n - 1) // 2)
    sampled = set()
    attempts = 0

    while len(sampled) < target and attempts < target * 3:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if (i != j and abs(i - j) > 1
                and (i, j) not in sampled and (j, i) not in sampled):
            sampled.add((i, j))
        attempts += 1

    for i, j in sampled:
        t1, t2 = indexed[i], indexed[j]
        sim = _cosine_sim(t1['embedding'], t2['embedding'])
        same_conv = t1['conv'] == t2['conv']
        same_sender = t1['sender'] == t2['sender']
        distance_norm = min(1.0, abs(i - j) / max(1, n))

        if not same_conv:
            if sim > 0.45:
                label, confidence = 'analogy', 0.4 + sim * 0.5
            elif sim > 0.25:
                label, confidence = 'metaphor', 0.3 + sim * 0.4
            else:
                label, confidence = 'symbol', 0.15 + sim * 0.3
        else:
            if not same_sender and sim > 0.45:
                label, confidence = 'resolution', 0.4 + sim * 0.3
            elif sim > 0.55:
                label, confidence = 'extension', 0.4 + sim * 0.4
            elif sim > 0.3:
                label, confidence = 'context', 0.2 + sim * 0.3
            else:
                label, confidence = 'contrast', 0.3 + (1.0 - sim) * 0.2

        confidence *= (1.0 - distance_norm * 0.3)
        confidence = max(0.1, min(1.0, confidence))

        pairs.append({
            'input': t1['embedding'] + t2['embedding'],
            'label': label,
            'confidence': confidence,
            'source': 'discovered',
            'depth_confidences': _depth_confidences(i, j, indexed),
        })

    return pairs


# ─── Text-Label Mining ────────────────────────────────────────────
#
# The corpus contains its own topology annotations embedded in natural
# language. When Tyler says "that's an analogy" or Claude says "that
# resolves the tension", those turns carry ground-truth labels.
# Mine them out. High confidence, source='text_mined'.

_TEXT_SIGNALS: Dict[str, List[str]] = {
    'analogy':    ['analog', 'similar to', 'resembles', 'parallel to',
                   'same as', 'mirrors this', 'maps to', 'equivalent to',
                   "it's like saying", 'think of it as'],
    'contrast':   ['in contrast', 'unlike', 'the opposite', 'however,',
                   'whereas ', 'on the other hand', 'differs from',
                   'not the same', 'flip side'],
    'extension':  ['extends', 'builds on', 'furthers', 'goes further',
                   'adds to', 'expands on', 'develops this', 'takes it further'],
    'resolution': ['resolves', 'that answers', 'settles', 'nailed it',
                   'you got it', 'that resolves', 'exactly right',
                   'that explains', 'so that means'],
    'metaphor':   ['metaphor', 'is a metaphor', 'think of it like',
                   'is really a', 'in the language of', 'as a metaphor'],
    'symbol':     ['symbol', 'signifies', 'points at something',
                   'ineffable', 'beyond words', 'gesture toward',
                   "can't say it directly"],
    'context':    ['for context', 'worth noting', 'to be clear,',
                   'relevant here', 'to situate', 'background:'],
}


def mine_text_labels(turns: List[Dict], max_mined: int = 1000) -> List[Dict]:
    """
    Scan each turn for explicit relational markers.
    When found, label the pair (turn[i-1], turn[i]) from the text.
    Returns list of high-confidence training pairs, source='text_mined'.
    """
    mined = []
    indexed = [t for t in turns if t.get('embedding')]

    for i in range(1, len(indexed)):
        t1, t2 = indexed[i - 1], indexed[i]
        if not (t1.get('embedding') and t2.get('embedding')):
            continue

        text = t2.get('text', '').lower()
        matched_label = None
        match_strength = 0.0

        for label, signals in _TEXT_SIGNALS.items():
            for sig in signals:
                if sig in text:
                    # earlier signals in list = stronger match
                    strength = 0.85 - signals.index(sig) * 0.03
                    if strength > match_strength:
                        matched_label = label
                        match_strength = strength

        if matched_label:
            dc = []
            for h in range(1, 4):
                k = i - 1 + h
                if k < len(indexed):
                    sim = _cosine_sim(indexed[i-1]['embedding'], indexed[k]['embedding'])
                    dc.append(max(0.05, sim))
            mined.append({
                'input':             t1['embedding'] + t2['embedding'],
                'label':             matched_label,
                'confidence':        round(max(0.6, match_strength), 3),
                'source':            'text_mined',
                'depth_confidences': dc if len(dc) >= 2 else None,
            })

        if len(mined) >= max_mined:
            break

    return mined


# ─── Label Balancing ──────────────────────────────────────────────

def balance_labels(pairs: List[Dict], max_share: float = 0.25) -> List[Dict]:
    """
    Cap any single label at max_share of total pairs.
    Text-mined pairs are always kept (they're ground truth).
    Positional pairs are downsampled like any other source if they overflow.
    Minority labels are upsampled (repeated with jitter) to reach at least
    floor_share of total pairs, so no label starves the MLP.
    """
    n_labels = len(LINK_TYPES)
    floor_share = 1.0 / (n_labels * 2)  # at least half of equal share

    # Separate text_mined (never touch) from everything else
    text_mined = [p for p in pairs if p.get('source') == 'text_mined']
    rest = [p for p in pairs if p.get('source') != 'text_mined']

    # Group rest by label
    by_label: Dict[str, List[Dict]] = {lt: [] for lt in LINK_TYPES}
    for p in rest:
        by_label.setdefault(p['label'], []).append(p)

    total_rest = len(rest)
    max_per_label = max(1, int(total_rest * max_share))

    # Downsample overrepresented labels
    capped = []
    for lt in LINK_TYPES:
        bucket = by_label[lt]
        if len(bucket) > max_per_label:
            bucket = random.sample(bucket, max_per_label)
        capped.extend(bucket)

    # Upsample underrepresented labels (repeat + small confidence jitter)
    total_after_cap = len(capped) + len(text_mined)
    floor_count = max(1, int(total_after_cap * floor_share))

    upsampled = []
    for lt in LINK_TYPES:
        bucket = [p for p in capped if p['label'] == lt]
        tm_bucket = [p for p in text_mined if p['label'] == lt]
        total_lt = len(bucket) + len(tm_bucket)
        if total_lt > 0 and total_lt < floor_count:
            need = floor_count - total_lt
            source = (bucket + tm_bucket) or bucket
            for _ in range(need):
                p = dict(random.choice(source))
                p['confidence'] = min(1.0, max(0.1, p['confidence'] + random.gauss(0, 0.02)))
                upsampled.append(p)

    balanced = capped + text_mined + upsampled
    random.shuffle(balanced)
    return balanced


# ─── Pair Compression (convergence-aware) ─────────────────────────

def compress_pairs(data: List[Dict], mlp, epoch: int,
                   keep_ratio: float = 0.7,
                   snr_history: Optional[List[float]] = None
                   ) -> Tuple[List[Dict], float]:
    if epoch < 5:
        return data, 1.0

    positional, boundary, collapsed = [], [], []

    for sample in data:
        if sample.get('source') in ('positional', 'text_mined'):
            positional.append(sample)
            continue

        x = np.array(sample['input'], dtype=np.float64)
        if len(x) < mlp.input_dim:
            x = np.pad(x, (0, mlp.input_dim - len(x)))
        elif len(x) > mlp.input_dim:
            x = x[:mlp.input_dim]

        probs, conf = mlp.forward(x)
        predicted = mlp.output_labels[np.argmax(probs)]
        correct = predicted == sample['label']
        top_prob = float(np.max(probs))

        if correct and conf > 0.85 and top_prob > 0.6:
            collapsed.append(sample)
        else:
            boundary.append(sample)

    snr = len(boundary) / max(1, len(collapsed)) if collapsed else float('inf')

    compression_ok = True
    if snr_history and len(snr_history) >= 2:
        prev_snr = snr_history[-1]
        if prev_snr > 0 and snr < prev_snr * 0.5:
            compression_ok = False

    decay = 0.8 if not compression_ok else max(0.2, keep_ratio ** (epoch / 10.0))
    n_keep = max(1, int(len(collapsed) * decay))
    kept = random.sample(collapsed, min(n_keep, len(collapsed)))

    return positional + boundary + kept, snr


# ─── Numpy MLP ────────────────────────────────────────────────────

class NumpyMLP:
    def __init__(self, input_dim, hidden_dims, output_labels,
                 lr=0.01, momentum=0.9):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_labels = output_labels
        self.output_dim = len(output_labels) + 1
        self.lr = lr
        self.lr_init = lr
        self.momentum = momentum

        dims = [input_dim] + hidden_dims + [self.output_dim]
        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
            self.weights.append(
                np.random.randn(dims[i], dims[i+1]).astype(np.float64) * scale)
            self.biases.append(np.zeros(dims[i+1], dtype=np.float64))
            self.vel_w.append(
                np.zeros((dims[i], dims[i+1]), dtype=np.float64))
            self.vel_b.append(np.zeros(dims[i+1], dtype=np.float64))

        self._acts = []
        self._pre = []

    def forward(self, x):
        self._acts = [x.copy()]
        self._pre = []
        current = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ W + b
            self._pre.append(z.copy())
            current = np.maximum(0.0, z) if i < len(self.weights) - 1 else z
            self._acts.append(current.copy())
        logits = current[:-1]
        raw_conf = current[-1]
        shifted = logits - logits.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()
        conf = 1.0 / (1.0 + np.exp(-np.clip(raw_conf, -500, 500)))
        return probs, conf

    def backward(self, target_label, target_confidence,
                 depth_confidences=None):
        output = self._acts[-1]
        logits = output[:-1]
        raw_conf = output[-1]
        shifted = logits - logits.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()
        conf = 1.0 / (1.0 + np.exp(-np.clip(raw_conf, -500, 500)))

        idx = (self.output_labels.index(target_label)
               if target_label in self.output_labels else 0)

        d_logits = probs.copy()
        d_logits[idx] -= 1.0
        conf_err = conf - target_confidence
        sig_grad = conf * (1.0 - conf)
        d_conf = 2.0 * conf_err * sig_grad

        a_loss = 0.0
        if depth_confidences and len(depth_confidences) >= 2:
            dc = depth_confidences
            td = 0.0
            for i in range(len(dc) - 1):
                r = max(1e-10, abs(dc[i])) / max(1e-10, abs(dc[i+1]))
                td += ((r - PHI) ** 2) * (PHI_INV ** i)
            a_loss = td / max(1, len(dc) - 1)
            cr = conf / max(1e-10, dc[-1])
            d_conf += 2.0 * (cr - PHI) * PHI_INV * sig_grad

        d_output = np.concatenate([d_logits, [d_conf]])
        ce = -np.log(max(1e-10, probs[idx]))
        mse = conf_err ** 2
        total_loss = ce + mse + a_loss

        d_current = d_output
        for i in range(len(self.weights) - 1, -1, -1):
            a_prev = self._acts[i]
            z = self._pre[i]
            if i < len(self.weights) - 1:
                d_current = d_current * (z > 0).astype(np.float64)
            d_W = np.outer(a_prev, d_current)
            d_b = d_current.copy()
            d_prev = self.weights[i] @ d_current
            self.vel_w[i] = self.momentum * self.vel_w[i] - self.lr * d_W
            self.weights[i] += self.vel_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] - self.lr * d_b
            self.biases[i] += self.vel_b[i]
            d_current = d_prev
        return total_loss

    def train_step(self, x, label, conf, depth_confs=None):
        self.forward(x)
        return self.backward(label, conf, depth_confs)

    def predict(self, x):
        probs, conf = self.forward(x)
        return self.output_labels[np.argmax(probs)], float(conf)

    def decay_lr(self, epoch, total):
        self.lr = self.lr_init * 0.5 * (1.0 + math.cos(math.pi * epoch / total))

    def save_weights(self, path):
        data = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_labels': self.output_labels,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_weights(self, path):
        with open(path) as f:
            data = json.load(f)
        self.weights = [np.array(w, dtype=np.float64) for w in data['weights']]
        self.biases  = [np.array(b, dtype=np.float64) for b in data['biases']]
        # reset velocity buffers to match loaded weight shapes
        self.vel_w = [np.zeros_like(w) for w in self.weights]
        self.vel_b = [np.zeros_like(b) for b in self.biases]


# ─── Training Loop ────────────────────────────────────────────────

def train(mlp, data, epochs=50, patience=10, verbose=True):
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    loss_history = []
    snr_history = []
    full_size = len(data)

    for epoch in range(epochs):
        active, snr = compress_pairs(data, mlp, epoch,
                                     snr_history=snr_history)
        snr_history.append(snr)
        random.shuffle(active)
        epoch_loss = 0.0

        for sample in active:
            x = np.array(sample['input'], dtype=np.float64)
            if len(x) < mlp.input_dim:
                x = np.pad(x, (0, mlp.input_dim - len(x)))
            elif len(x) > mlp.input_dim:
                x = x[:mlp.input_dim]
            epoch_loss += mlp.train_step(
                x, sample['label'], sample['confidence'],
                sample.get('depth_confidences'))

        avg = epoch_loss / max(1, len(active))
        loss_history.append(avg)
        mlp.decay_lr(epoch, epochs)

        if avg < best_loss:
            best_loss = avg
            best_weights = ([w.copy() for w in mlp.weights],
                           [b.copy() for b in mlp.biases])
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1
                        or no_improve >= patience):
            comp = 1.0 - len(active) / full_size if full_size else 0
            s = f"{snr:.2f}" if snr != float('inf') else "inf"
            print(f"  epoch {epoch:3d}  loss={avg:.4f}  best={best_loss:.4f}  "
                  f"pairs={len(active)}/{full_size} ({comp:.0%} compressed)  "
                  f"snr={s}  lr={mlp.lr:.5f}")
            sys.stdout.flush()

        if no_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_weights:
        mlp.weights, mlp.biases = best_weights

    return {
        'epochs': len(loss_history),
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'best_loss': best_loss,
        'status': 'converged' if no_improve >= patience else 'completed',
        'final_snr': snr_history[-1] if snr_history else 0.0,
    }


# ─── Main ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="JANUS Link MLP — portable training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=TOTAL_CORPUS)
    parser.add_argument("--max-pairs", type=int, default=10000)
    parser.add_argument("--no-stop", action="store_true", help="disable early stopping, run all epochs")
    parser.add_argument("--fresh", action="store_true",
                        help="Reset checkpoint (cache still used)")
    parser.add_argument("--backend", choices=[
        "sentence_transformers", "ollama", "voyage"],
        default=None, help="Override EMBED_BACKEND")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible initialization and pair sampling")
    parser.add_argument("--warmstart", action="store_true",
                        help="Load existing weights from link_mlp_trained.json and continue training")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"  seed: {args.seed}")

    global EMBED_BACKEND
    if args.backend:
        EMBED_BACKEND = args.backend

    print("=" * 64)
    print("  JANUS Link MLP — Portable Training")
    print(f"  backend: {EMBED_BACKEND}  dims: {TARGET_DIM}×2=128")
    print("=" * 64)
    sys.stdout.flush()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache()
    ckpt = EmbedCheckpoint()
    if args.fresh:
        ckpt.reset()

    print(f"\n  cache: {len(cache)} vectors")
    print(f"  frontier: {ckpt.total_embedded} embedded")
    sys.stdout.flush()

    # Embed
    print(f"\nLoading corpus (max {args.max_turns} turns)...")
    sys.stdout.flush()
    t0 = time.time()
    turns = load_and_embed_turns(args.max_turns, cache, ckpt)
    elapsed = time.time() - t0
    print(f"  {len(turns)} turns in {elapsed:.1f}s  "
          f"(cache: {len(cache)} vectors)")

    if len(turns) < 4:
        print("Not enough turns.")
        sys.exit(1)

    # Links + pairs
    turns = add_positional_links(turns)
    n_links = sum(1 for t in turns if t.get("link_to_next"))
    print(f"  positional links: {n_links}")

    if args.warmstart and PAIRS_PATH.exists():
        print("\nLoading saved training pairs...")
        with open(PAIRS_PATH) as f:
            data = json.load(f)
        print(f"  pairs: {len(data)}  (from {PAIRS_PATH.name})")
    else:
        print("\nGenerating training pairs...")
        data = generate_training_pairs(turns, max_pairs=args.max_pairs)
        print(f"  pairs: {len(data)}")

        print("\nMining text labels from corpus...")
        text_pairs = mine_text_labels(turns)
        data = data + text_pairs
        print(f"  text-mined pairs: {len(text_pairs)}  (total: {len(data)})")

        print("\nBalancing label distribution...")
        data = balance_labels(data)

        print(f"\nSaving training pairs → {PAIRS_PATH.name}")
        with open(PAIRS_PATH, 'w') as f:
            json.dump(data, f)

    if not data:
        print("No training data.")
        sys.exit(1)

    input_dim = len(data[0]["input"])
    print(f"  input_dim: {input_dim}")

    # Label distribution
    lc = Counter(d['label'] for d in data)
    sc = Counter(d['source'] for d in data)
    print(f"\n  label distribution:")
    for lt in LINK_TYPES:
        c = lc.get(lt, 0)
        p = c / len(data) * 100
        bar = '█' * int(p / 2) + '░' * (50 - int(p / 2))
        ok = '✓' if c > 0 else '✗'
        print(f"    {ok} {lt:<12} {c:>6}  {p:>5.1f}%  {bar}")
    print(f"    sources: {dict(sc)}")
    dead = [lt for lt in LINK_TYPES if lc.get(lt, 0) == 0]
    if dead:
        print(f"    WARNING: dead labels {dead}")

    # Train
    _init_lr = 0.001 if args.warmstart else 0.01
    print(f"\nNumpyMLP: {input_dim} → [64, 32] → {len(LINK_TYPES)+1}")
    mlp = NumpyMLP(input_dim, [64, 32], LINK_TYPES, lr=_init_lr, momentum=0.9)

    if args.warmstart and WEIGHTS_PATH.exists():
        mlp.load_weights(str(WEIGHTS_PATH))
        print(f"  warmstart: loaded weights from {WEIGHTS_PATH.name}  (lr={_init_lr})")
    elif args.warmstart:
        print(f"  warmstart: no weights found at {WEIGHTS_PATH.name}, starting fresh")

    _patience = 9999 if args.no_stop else 10
    print(f"\nTraining (up to {args.epochs} epochs, patience={_patience})...\n")
    sys.stdout.flush()
    stats = train(mlp, data, epochs=args.epochs, patience=_patience)

    print(f"\n  status:     {stats['status']}")
    print(f"  epochs:     {stats['epochs']}")
    print(f"  best_loss:  {stats['best_loss']:.4f}")
    print(f"  final_loss: {stats['final_loss']:.4f}")

    # Save
    print(f"\n  saving backup → {BACKUP_PATH.name}")
    mlp.save_weights(str(BACKUP_PATH))
    print(f"  saving weights → {WEIGHTS_PATH.name}")
    mlp.save_weights(str(WEIGHTS_PATH))

    with open(WEIGHTS_PATH) as f:
        saved = json.load(f)
    print(f"  verified: input_dim={saved['input_dim']}  "
          f"labels={saved['output_labels']}")

    x = np.array(data[0]['input'][:128], dtype=np.float64)
    label, conf = mlp.predict(x)
    print(f"  sanity: '{label}' @ {conf:.3f}")

    remaining = TOTAL_CORPUS - len(cache)
    print(f"\n{'=' * 64}")
    print(f"  Cache: {len(cache)}/{TOTAL_CORPUS} "
          f"({len(cache)/TOTAL_CORPUS:.0%} embedded)")
    if remaining > 0:
        print(f"  Remaining: {remaining} turns")
    else:
        print(f"  Full corpus embedded.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
