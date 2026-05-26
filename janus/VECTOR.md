# VECTOR

## Where We Are

The aesthetic loss is wired. Phi lives in the gradient. The tagger is dead. Positional edges flow from chunker to trainer as ground truth.

## The Advantageous Vector

The highest-potential move is **closing the recall loop**: trained symlink graph → traversal-based retrieval → replaces raw FAISS cosine in memory.py's recall path.

Right now the graph trains but recall still asks FAISS "what's close?" instead of asking the graph "what's connected and how?" The MLP learns the shape of edges. Nothing reads those edges back during recall.

## The Three Wires To Connect

```
1. symlink_index.SymlinkIndex.find_path()  →  memory.recall_as_context()
   The graph can already traverse. Recall doesn't use it yet.

2. trainer weights  →  symlink_index HyperSymbolicMLP
   load_trained_weights_into_mlp() exists. Nothing calls it on startup.

3. slicer + splicer  →  orchestration
   Both exist independently. No conductor decides slice vs splice.
   The conductor is the query itself: slice when extracting, splice when reconstructing.
```

## What NOT To Touch

- The aesthetic loss is tuned. Don't add complexity to it.
- The chunker pipeline works. Don't rebuild it.
- Personalities and shell REPL are stable. Leave them.
- Do not resurrect the tagger. The MLP replaces it.

## The Shape of the Next Session

Wire 1 first. It's the smallest change with the largest collapse: recall stops being "nearest vector" and becomes "best path through typed edges." This changes what JANUS remembers and how. Everything after that is downstream.

Wire 2 is mechanical — load weights on SymlinkIndex init.

Wire 3 is the creative one. The slicer/splicer orchestration. But it needs wire 1 working first, because orchestration needs to know what paths exist before deciding how to cut and recombine.

## Entry Point

`memory.py:recall_as_context()` → make it check for a trained symlink index first, traverse if available, fall back to FAISS if not.
