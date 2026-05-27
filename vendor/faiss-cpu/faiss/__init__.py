"""Numpy shim for the faiss API surface used by bud-rag.

Implements IndexFlatIP (cosine similarity via inner product after L2 normalisation),
normalize_L2, write_index, and read_index — enough to replace faiss-cpu on platforms
where it cannot be built (e.g. Android/Termux aarch64).

read_index transparently handles both:
  - Files written by the real faiss library (IxFI binary format)
  - Files written by this shim (numpy binary format)
"""

import struct

import numpy as np

# fourcc("IxFI") little-endian — identifies a real faiss IndexFlatIP file
_FAISS_MAGIC = struct.unpack("<I", b"IxFI")[0]
# numpy .npy magic — identifies a file written by this shim
_NPY_MAGIC = b"\x93NUMPY"
# 45-byte FAISS flat-index header: fourcc I, d i, ntotal q, dummy q, dummy q, is_trained B, metric_type i, float_count Q
_FAISS_HDR = struct.Struct("<IiqqqBiQ")


def normalize_L2(arr: np.ndarray) -> None:
    """Normalise rows of arr to unit L2 norm, in-place."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arr /= norms


class IndexFlatIP:
    """Flat inner-product index (exhaustive cosine search after L2 normalisation)."""

    def __init__(self, dim: int):
        self.d = dim
        self._vecs: np.ndarray | None = None

    @property
    def ntotal(self) -> int:
        return len(self._vecs) if self._vecs is not None else 0

    def add(self, arr: np.ndarray) -> None:
        self._vecs = arr.copy() if self._vecs is None else np.vstack([self._vecs, arr])

    def search(self, arr: np.ndarray, k: int):
        if self._vecs is None or len(self._vecs) == 0:
            return (
                np.full((1, k), -1.0, dtype=np.float32),
                np.full((1, k), -1, dtype=np.int64),
            )
        scores = (arr @ self._vecs.T)[0]
        k = min(k, len(scores))
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return scores[top].reshape(1, -1), top.reshape(1, -1).astype(np.int64)


def write_index(index: IndexFlatIP, path: str) -> None:
    """Write index in numpy format (portable, no native faiss required)."""
    vecs = index._vecs if index._vecs is not None else np.empty((0, index.d), dtype=np.float32)
    with open(path, "wb") as f:
        np.save(f, vecs)
        np.save(f, np.array([index.d], dtype=np.int64))


def _read_faiss_binary(path: str) -> "IndexFlatIP | None":
    """Parse an IndexFlatIP file written by the real faiss library.

    Empirically-verified format (45-byte header):
      uint32  magic    = fourcc("IxFI")
      int32   d        (dimension)
      int64   ntotal   (vector count)
      int64   dummy    (1<<20, written twice for historical reasons)
      int64   dummy
      uint8   is_trained
      int32   metric_type
      uint64  float_count = ntotal * d
    Followed by float_count * 4 bytes of raw float32 vectors.
    """
    hdr_size = _FAISS_HDR.size  # 45
    with open(path, "rb") as f:
        hdr_bytes = f.read(hdr_size)
        if len(hdr_bytes) < hdr_size:
            return None
        magic, d, ntotal, _, _, _, _, float_count = _FAISS_HDR.unpack(hdr_bytes)
        if magic != _FAISS_MAGIC:
            return None
        if ntotal == 0:
            return IndexFlatIP(d)
        expected = ntotal * d
        if float_count != expected:
            return None
        raw = f.read(expected * 4)
    vecs = np.frombuffer(raw, dtype=np.float32).reshape(ntotal, d).copy()
    idx = IndexFlatIP(d)
    idx._vecs = vecs
    return idx


def read_index(path: str) -> IndexFlatIP:
    """Read an index file, auto-detecting real-faiss binary or numpy shim format."""
    with open(path, "rb") as f:
        probe = f.read(5)

    if probe[:4] == struct.pack("<I", _FAISS_MAGIC)[:4]:
        result = _read_faiss_binary(path)
        if result is not None:
            return result

    # numpy shim format
    with open(path, "rb") as f:
        vecs = np.load(f, allow_pickle=False)
        dim = int(np.load(f, allow_pickle=False)[0])
    idx = IndexFlatIP(dim)
    if len(vecs) > 0:
        idx._vecs = vecs
    return idx
