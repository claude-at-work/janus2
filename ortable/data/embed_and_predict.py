"""
Quick script: embed conversation turns from the JSONL files,
reduce to 64-dim, pair them, and run through the trained link MLP.
"""
import json
import sys
import math
import urllib.request
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "all-minilm:33m"  # 384-dim, fast
MLP_PATH = Path(__file__).parent.parent / "link_mlp_trained.json"
DATA_DIR = Path(__file__).parent
MAX_TURNS = 40  # keep it quick


# ─── Embedding ────────────────────────────────────────────────────
def embed_text(text: str) -> list:
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        embs = data.get("embeddings", [])
        return embs[0] if embs else []


def reduce_dim(vec: list, target: int = 64) -> list:
    """Stride-sample down to target dimensions."""
    if len(vec) <= target:
        return vec + [0.0] * (target - len(vec))
    step = len(vec) / target
    return [vec[int(i * step)] for i in range(target)]


# ─── MLP inference ────────────────────────────────────────────────
def load_mlp(path):
    with open(path) as f:
        return json.load(f)


def relu(x):
    return max(0.0, x)


def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def mlp_forward(mlp, input_vec):
    """Forward pass through the trained MLP. Returns (label, probs, confidence)."""
    x = input_vec[:]
    weights = mlp["weights"]
    biases = mlp["biases"]
    labels = mlp["output_labels"]

    for i, (W, b) in enumerate(zip(weights, biases)):
        y = []
        for j in range(len(b)):
            val = b[j]
            for k in range(len(x)):
                val += x[k] * W[k][j]
            if i < len(weights) - 1:
                val = relu(val)
            y.append(val)
        x = y

    probs = softmax(x[:-1])
    confidence = 1.0 / (1.0 + math.exp(-max(-500, min(500, x[-1]))))

    best_idx = probs.index(max(probs))
    return labels[best_idx], probs, confidence


# ─── Load conversations ──────────────────────────────────────────
def load_turns(max_turns=MAX_TURNS):
    turns = []
    for jf in sorted(DATA_DIR.glob("conversations_*.jsonl")):
        with open(jf) as f:
            for line in f:
                rec = json.loads(line)
                for t in rec.get("turns", []):
                    text = t.get("text", "").strip()
                    if text and len(text) > 20:
                        turns.append({
                            "text": text[:512],  # truncate for embedding speed
                            "sender": t.get("sender", "?"),
                            "conv": rec.get("conversation_name", ""),
                        })
                        if len(turns) >= max_turns:
                            return turns
    return turns


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("Loading MLP weights...")
    mlp = load_mlp(MLP_PATH)
    print(f"  input_dim={mlp['input_dim']}  labels={mlp['output_labels']}")

    print(f"\nLoading up to {MAX_TURNS} turns from conversations...")
    turns = load_turns()
    print(f"  got {len(turns)} turns")

    if len(turns) < 2:
        print("Need at least 2 turns to form pairs.")
        return

    print(f"\nEmbedding {len(turns)} turns with {EMBED_MODEL}...")
    for i, t in enumerate(turns):
        t["embedding"] = reduce_dim(embed_text(t["text"]))
        sys.stdout.write(f"\r  embedded {i+1}/{len(turns)}")
        sys.stdout.flush()
    print()

    # Pair adjacent turns and some non-adjacent
    pairs = []
    # Adjacent
    for i in range(len(turns) - 1):
        pairs.append((i, i + 1, "adjacent"))
    # Skip-1
    for i in range(len(turns) - 2):
        pairs.append((i, i + 2, "skip-1"))
    # A few distant
    for i in range(0, len(turns) - 5, 5):
        pairs.append((i, i + 5, "distant"))

    print(f"\nRunning {len(pairs)} pairs through MLP...\n")
    print(f"{'Pair':<12} {'Link Type':<14} {'Conf':>6}  {'Probs'}")
    print("─" * 72)

    for i, j, kind in pairs:
        inp = turns[i]["embedding"] + turns[j]["embedding"]
        label, probs, conf = mlp_forward(mlp, inp)
        prob_str = "  ".join(f"{mlp['output_labels'][k]}:{p:.2f}" for k, p in enumerate(probs))
        t1 = turns[i]["text"][:30].replace("\n", " ")
        t2 = turns[j]["text"][:30].replace("\n", " ")
        print(f"[{kind:<8}]  {label:<14} {conf:.3f}  {prob_str}")
        if kind == "adjacent":
            print(f"            {t1}...")
            print(f"         →  {t2}...")
            print()


if __name__ == "__main__":
    main()
