"""
JANUS Symlink Index -- building a language from symbolic links.

The core idea: Instead of just embedding text, we build a graph of symbolic
links that create pathways through the knowledge space. Each chunk gets
symbolic link tags that form connections to other chunks.

This creates a "language" where meaning is found through traversal paths,
not just vector similarity.

-- HYPER-SYMBOLIC EXTENSION --

Lossy hyper-symbolic links use MLPs to learn relationships between chunks.
The "lossy" aspect is intentional - we trade precision for flexibility,
allowing approximate, probabilistic relationships that capture semantic
similarities beyond strict schema matching.

The MLP takes chunk embeddings + schema tags as input and outputs:
- Link type probabilities
- Link strength (confidence)
- Multi-level symbolic representations
"""

import os
import json
import hashlib
import re
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import datetime


# ─── Symbolic Link Structure ────────────────────────────────────


@dataclass
class SymbolicLink:
    """
    A symbolic link between chunks that creates meaning through relationship.

    Instead of "closest vector", we have "path through links".

    HYPER-SYMBOLIC EXTENSION:
    - lossy: probabilistic, approximate relationships
    - hyper: multi-level symbolic representations
    - mlp: learned from embeddings + schema tags
    """
    source_chunk_id: str          # The chunk this link originates from
    target_chunk_id: str          # The chunk this link points to
    link_type: str                # Type of relationship (e.g., "analogy", "contrast", "extension")
    link_value: float             # Strength/direction of link (-1 to 1)
    link_symbols: str             # Visual representation (e.g., "→", "↔", "≈")
    link_confidence: float = 1.0  # MLP confidence score (0-1)
    lossy: bool = False           # Whether this is a lossy (approximate) link
    hyper_depth: int = 1          # Hyper-symbolic depth (1 = base, 2+ = meta-links)
    timestamp: float = field(default_factory=time.time)

    @property
    def hash(self) -> str:
        """Unique hash for this link."""
        content = f"{self.source_chunk_id}:{self.target_chunk_id}:{self.link_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    """
    A chunk with symbolic links attached.

    This is how we build the language: chunks aren't isolated,
    they're nodes in a web of relationships.

    HYPER-SYMBOLIC: Chunks can have multiple levels of symbolic meaning.
    """
    chunk_id: str
    text: str
    embedding: List[float]
    links: List[SymbolicLink] = field(default_factory=list)
    schema_tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_link(self, target_id: str, link_type: str, value: float, symbols: str,
                 confidence: float = 1.0, lossy: bool = False, hyper_depth: int = 1):
        """Add a symbolic link to another chunk."""
        link = SymbolicLink(
            source_chunk_id=self.chunk_id,
            target_chunk_id=target_id,
            link_type=link_type,
            link_value=value,
            link_symbols=symbols,
            link_confidence=confidence,
            lossy=lossy,
            hyper_depth=hyper_depth
        )
        self.links.append(link)
        return link

    def get_paths_to(self, target_id: str, max_depth: int = 3,
                     min_confidence: float = 0.0) -> List[List[SymbolicLink]]:
        """
        Find paths from this chunk to another through symbolic links.

        This is how we answer queries: not by vector similarity,
        but by traversing the relationship graph.

        HYPER-SYMBOLIC: Can filter by confidence and hyper-depth.
        """
        paths = []

        def search(current: str, path: List[SymbolicLink], visited: Set[str], depth: int):
            if depth > max_depth:
                return
            if current == target_id:
                paths.append(path.copy())
                return

            # Find links from current chunk
            for link in self.links:
                if link.source_chunk_id == current:
                    if link.target_chunk_id not in visited:
                        if link.link_confidence >= min_confidence:
                            visited.add(link.target_chunk_id)
                            path.append(link)
                            search(link.target_chunk_id, path, visited, depth + 1)
                            path.pop()
                            visited.remove(link.target_chunk_id)

        visited = {self.chunk_id}
        search(self.chunk_id, [], visited, 0)
        return paths


# ─── Simple MLP for Link Generation ─────────────────────────────


class SimpleMLP:
    """
    Simple feed-forward MLP for learning symbolic link relationships.

    Architecture:
    Input: [chunk1_embedding, chunk2_embedding, schema1_onehot, schema2_onehot]
    Hidden: [64, 32]
    Output: [link_type_probs, confidence]

    HYPER-SYMBOLIC: Multiple MLPs can be stacked for hierarchical reasoning.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 link_types: List[str] = None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.link_types = link_types or ['analogy', 'contrast', 'extension', 'context']

        # Initialize weights (Xavier initialization)
        self.weights = []
        self.biases = []

        layer_dims = [input_dim] + self.hidden_dims + [len(self.link_types) + 1]

        for i in range(len(layer_dims) - 1):
            # Xavier/Glorot initialization
            scale = math.sqrt(2.0 / (layer_dims[i] + layer_dims[i + 1]))
            layer_weights = [[(hash(f"w_{i}_{j}_{k}") % 100) / 100.0 * scale
                              for k in range(layer_dims[i + 1])]
                             for j in range(layer_dims[i])]
            layer_biases = [(hash(f"b_{i}_{j}") % 100) / 100.0
                            for j in range(layer_dims[i + 1])]
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

    def _relu(self, x: float) -> float:
        """ReLU activation."""
        return max(0.0, x)

    def _softmax(self, logits: List[float]) -> List[float]:
        """Softmax for probability distribution."""
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        return [e / total for e in exp_logits]

    def forward(self, input_vec: List[float]) -> Tuple[List[float], float]:
        """
        Forward pass through the MLP.

        Returns:
            link_type_probs: Probability distribution over link types
            confidence: Overall confidence score (0-1)
        """
        x = input_vec[:]

        for i, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            # Linear layer: y = x @ W + b
            next_len = len(bias)
            y = []
            for j in range(next_len):
                val = bias[j]
                for k in range(len(x)):
                    val += x[k] * weights[k][j]
                # Apply activation (ReLU for hidden, no activation for output)
                if i < len(self.weights) - 1:
                    val = self._relu(val)
                y.append(val)

            x = y

        # Split output: link type probabilities + confidence
        link_probs = self._softmax(x[:-1])
        confidence = min(1.0, max(0.0, x[-1]))  # Sigmoid approximation

        return link_probs, confidence

    def predict(self, input_vec: List[float]) -> Tuple[str, float]:
        """
        Predict link type and confidence.

        Returns:
            link_type: Most likely link type
            confidence: Confidence score
        """
        probs, conf = self.forward(input_vec)
        idx = probs.index(max(probs))
        return self.link_types[idx], conf


# ─── Slicing MLP - Extracts source chunks ───────────────────────


class SlicingMLP:
    """
    MLP to predict which chunks to slice (extract).

    Input: [chunk_text_blob, chunk_embedding]
    Output: [slice_prob, importance_score]

    The "blob" is a simplified representation of the chunk content.
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim + 32  # embedding + simplified text blob
        self.output_dim = 2  # slice_prob, importance

        # Initialize weights
        self.W1 = [[(hash(f"W1_{i}_{j}") % 100) / 100.0 - 0.5
                    for j in range(64)] for i in range(self.input_dim)]
        self.b1 = [(hash(f"b1_{i}") % 100) / 100.0 - 0.5 for i in range(64)]

        self.W2 = [[(hash(f"W2_{i}_{j}") % 100) / 100.0 - 0.5
                    for j in range(self.output_dim)] for i in range(64)]
        self.b2 = [(hash(f"b2_{i}") % 100) / 100.0 - 0.5 for i in range(self.output_dim)]

    def _relu(self, x):
        return max(0.0, x)

    def predict_slice(self, query_emb: List[float], chunk_emb: List[float],
                      schema_tags: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Predict if chunk should be sliced and its importance.

        Input: query embedding, chunk embedding, optional schema tags
        Output: (slice_prob, importance_score)
        """
        # Build a 32-dim feature from query embedding + schema tag hashing
        if schema_tags:
            tag_str = json.dumps(schema_tags, sort_keys=True)
        else:
            tag_str = ""
        blob_encoding = [(hash(c) % 100) / 100.0 for c in tag_str[:32]]
        blob_encoding = blob_encoding + [0] * (32 - len(blob_encoding))

        # Use difference between query and chunk embeddings as signal
        min_len = min(len(query_emb), len(chunk_emb), self.embedding_dim)
        diff_emb = [query_emb[i] - chunk_emb[i] if i < min_len else 0.0 for i in range(self.embedding_dim)]
        emb = diff_emb

        input_vec = blob_encoding + emb

        # Forward pass
        h = []
        for j in range(64):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += input_vec[i] * self.W1[i][j]
            h.append(self._relu(val))

        output = []
        for j in range(self.output_dim):
            val = self.b2[j]
            for i in range(64):
                val += h[i] * self.W2[i][j]
            output.append(val)

        # Sigmoid for probabilities
        import math
        slice_prob = 1.0 / (1.0 + math.exp(-output[0]))
        importance = 1.0 / (1.0 + math.exp(-output[1]))

        return slice_prob, importance


# ─── Splicing MLP - Reconstructs from blob ──────────────────────


class BlobSplicingMLP:
    """
    MLP to reconstruct/splice text from sliced chunks using raw blob input.

    Input: [slices_blob, context_embedding]
    Output: {reconstruction, splice_type, confidence, weights}
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.input_dim = 96 + embedding_dim * 2
        self.output_dim = embedding_dim

        self.W1 = [[(hash(f"S1_{i}_{j}") % 100) / 100.0 - 0.5
                    for j in range(64)] for i in range(self.input_dim)]
        self.b1 = [(hash(f"Sb1_{i}") % 100) / 100.0 - 0.5 for i in range(64)]

        self.W2 = [[(hash(f"S2_{i}_{j}") % 100) / 100.0 - 0.5
                    for j in range(32)] for i in range(64)]
        self.b2 = [(hash(f"Sb2_{i}") % 100) / 100.0 - 0.5 for i in range(32)]

        self.W3 = [[(hash(f"S3_{i}_{j}") % 100) / 100.0 - 0.5
                    for j in range(self.output_dim)] for i in range(32)]
        self.b3 = [(hash(f"Sb3_{i}") % 100) / 100.0 - 0.5 for i in range(self.output_dim)]

        self.W_class = [[(hash(f"Sc_{i}_{j}") % 100) / 100.0 - 0.5
                        for j in range(4)] for i in range(64)]
        self.b_class = [(hash(f"Sbc_{i}") % 100) / 100.0 - 0.5 for i in range(4)]

    def _relu(self, x):
        return max(0.0, x)

    def _softmax(self, logits):
        max_logit = max(logits)
        exp = [math.exp(l - max_logit) for l in logits]
        total = sum(exp)
        return [e / total for e in exp]

    def predict_splice(self, slices_blob: List[str], context_emb: List[float]) -> Dict[str, Any]:
        """Predict how to splice raw text blobs together."""
        blob_encoding = []
        for s in slices_blob[:3]:
            encoded = [(hash(c) % 100) / 100.0 for c in s[:32]]
            blob_encoding.extend(encoded)
        blob_encoding = blob_encoding + [0] * (96 - len(blob_encoding))

        ctx_emb = context_emb[:self.embedding_dim] if len(context_emb) >= self.embedding_dim else context_emb + [0] * (self.embedding_dim - len(context_emb))

        input_vec = blob_encoding + ctx_emb

        h1 = []
        for j in range(64):
            val = self.b1[j]
            for i in range(self.input_dim):
                val += input_vec[i] * self.W1[i][j]
            h1.append(self._relu(val))

        h2 = []
        for j in range(32):
            val = self.b2[j]
            for i in range(64):
                val += h1[i] * self.W2[i][j]
            h2.append(self._relu(val))

        reconstruction = []
        for j in range(self.output_dim):
            val = self.b3[j]
            for i in range(32):
                val += h2[i] * self.W3[i][j]
            reconstruction.append(val)

        class_logits = []
        for j in range(4):
            val = self.b_class[j]
            for i in range(64):
                val += h1[i] * self.W_class[i][j]
            class_logits.append(val)
        class_probs = self._softmax(class_logits)

        splice_types = ['concat', 'analogical', 'contrastive', 'sequential']
        splice_type = splice_types[class_probs.index(max(class_probs))]

        num_slices = len(slices_blob)
        if num_slices == 1:
            weights = {'primary': 1.0}
        elif num_slices == 2:
            weights = {'primary': 0.5, 'secondary': 0.5}
        else:
            weights = {'primary': 0.4, 'secondary': 0.35, 'context': 0.25}

        return {
            'reconstruction': reconstruction,
            'splice_type': splice_type,
            'confidence': max(class_probs),
            'weights': weights
        }


# ─── Splicing MLP - Combines slices via embedding comparison ────


class SplicingMLP(SimpleMLP):
    """
    MLP to predict how two chunks should be spliced based on their
    embeddings and link type. Inherits SimpleMLP for forward/predict.

    Input: [chunk1_emb, chunk2_emb, link_type_onehot]
    Output: (splice_type, confidence, position_weights)
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim * 2 + 8

        super().__init__(self.input_dim, hidden_dims=[64, 32], link_types=[
            'concat', 'analogical', 'contrastive', 'contextual',
            'sequential', 'hierarchical'
        ])

    def predict_splice(self, chunk1_emb: List[float], chunk2_emb: List[float],
                       link_type: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict splice type and weights from two chunk embeddings.

        Returns:
            splice_type: How to combine chunks
            confidence: How confident in prediction
            weights: Position weights for each chunk
        """
        link_types = ['analogy', 'contrast', 'extension', 'context',
                      'meta_analogy', 'meta_contrast', 'pattern', 'emergence']
        link_encoding = [1.0 if link_type == lt else 0.0 for lt in link_types]

        emb1 = chunk1_emb[:self.embedding_dim] if len(chunk1_emb) >= self.embedding_dim else chunk1_emb + [0] * (self.embedding_dim - len(chunk1_emb))
        emb2 = chunk2_emb[:self.embedding_dim] if len(chunk2_emb) >= self.embedding_dim else chunk2_emb + [0] * (self.embedding_dim - len(chunk2_emb))

        input_vec = emb1 + emb2 + link_encoding
        splice_type, confidence = self.predict(input_vec)

        if splice_type == 'analogical':
            weights = {'chunk1': 0.4, 'chunk2': 0.4, 'connector': 0.2}
        elif splice_type == 'contrastive':
            weights = {'chunk1': 0.6, 'chunk2': 0.6, 'connector': 0.2}
        elif splice_type == 'hierarchical':
            weights = {'chunk1': 0.3, 'chunk2': 0.7, 'connector': 0.2}
        else:
            weights = {'chunk1': 0.5, 'chunk2': 0.5, 'connector': 0.2}

        return splice_type, confidence, weights


# ─── Hyper-Symbolic MLP ─────────────────────────────────────────
class HyperSymbolicMLP:
    """
    Multi-level MLP for hierarchical hyper-symbolic link generation.

    Level 1: Base links (analogy, contrast, extension)
    Level 2: Meta-links (relations between relations)
    Level 3: Abstract patterns (high-order abstractions)

    HYPER-SYMBOLIC: Each level can use different MLP configurations.
    """

    def __init__(self, embedding_dim: int = 384, schema_dims: int = 16):
        self.embedding_dim = embedding_dim
        self.schema_dims = schema_dims
        self.input_dim = embedding_dim * 2 + schema_dims * 2

        # MLPs for different hyper depths
        self.mlps = {
            1: SimpleMLP(self.input_dim, link_types=[
                'analogy', 'contrast', 'extension', 'context',
                'resolution', 'metaphor', 'symbol'
            ]),
            2: SimpleMLP(self.input_dim + 8, link_types=[
                'meta_analogy', 'meta_contrast', 'meta_extension'
            ]),
            3: SimpleMLP(self.input_dim + 16, link_types=[
                'pattern', 'abstraction', 'emergence'
            ])
        }

    def generate_input(self, emb1: List[float], emb2: List[float],
                       tags1: Dict, tags2: Dict) -> List[float]:
        """Generate combined input vector from embeddings and tags."""
        # Normalize embeddings
        emb1_norm = emb1[:self.embedding_dim] if len(emb1) >= self.embedding_dim else emb1 + [0] * (self.embedding_dim - len(emb1))
        emb2_norm = emb2[:self.embedding_dim] if len(emb2) >= self.embedding_dim else emb2 + [0] * (self.embedding_dim - len(emb2))

        # One-hot encode schema tags
        schema1 = []
        schema2 = []

        for dim in ['geometry', 'texture', 'terrain', 'coherence']:
            val1 = tags1.get(dim, '')
            val2 = tags2.get(dim, '')
            all_tags = ['linear', 'recursive', 'spiral', 'bifurcating',
                       'dense', 'sparse', 'lyrical', 'technical',
                       'conceptual', 'emotional', 'procedural', 'speculative',
                       'tight', 'loose', 'fragmented', 'emergent']
            schema1.append(1.0 if val1 in all_tags else 0.0)
            schema2.append(1.0 if val2 in all_tags else 0.0)

        # Pad schema
        schema1 = schema1[:self.schema_dims] + [0] * (self.schema_dims - len(schema1))
        schema2 = schema2[:self.schema_dims] + [0] * (self.schema_dims - len(schema2))

        return emb1_norm + emb2_norm + schema1 + schema2

    def generate_links(self, emb1: List[float], emb2: List[float],
                       tags1: Dict, tags2: Dict,
                       max_depth: int = 2) -> List[SymbolicLink]:
        """
        Generate hyper-symbolic links using MLP at each depth level.

        Returns:
            List of SymbolicLink with various hyper_depth values
        """
        links = []
        input_vec = self.generate_input(emb1, emb2, tags1, tags2)

        for depth in range(1, max_depth + 1):
            if depth not in self.mlps:
                break

            mlp = self.mlps[depth]
            link_type, confidence = mlp.predict(input_vec)

            # Determine symbols based on link type
            symbols_map = {
                'analogy': '→', 'contrast': '↔', 'extension': '⇒',
                'context': '≈', 'resolution': '✓', 'metaphor': '%%',
                'symbol': '∞', 'meta_analogy': '≡', 'meta_contrast': '≢',
                'meta_extension': '≣', 'pattern': '∞', 'abstraction': '∑',
                'emergence': '∆'
            }
            symbols = symbols_map.get(link_type, '≈')

            # Lossy at higher depths (more approximate)
            lossy = depth > 1

            link = SymbolicLink(
                source_chunk_id="",
                target_chunk_id="",
                link_type=link_type,
                link_value=confidence * 0.8,  # Scale confidence to link_value range
                link_symbols=symbols,
                link_confidence=confidence,
                lossy=lossy,
                hyper_depth=depth
            )
            links.append(link)

        return links


# ─── Symlink Index with Hyper-Symbolic Support ──────────────────


class SymlinkIndex:
    """
    Index built on symbolic links rather than just vectors.

    The architecture:
    1. Chunk text as usual
    2. Generate embeddings as usual
    3. BUT also generate symbolic links between chunks
    4. Store both vectors AND link graph

    Query becomes: Find paths through link graph, then check vector similarity.

    -- HYPER-SYMBOLIC EXTENSION --

    The index now supports:
    - Lossy links: Probabilistic relationships with confidence scores
    - Hyper-symbolic links: Multi-level symbolic representations
    - MLP-generated links: Learned relationships from embeddings + tags
    """

    def __init__(self, index_path: str,
                 use_mlp: bool = True,
                 hyper_depth: int = 2,
                 lossy_threshold: float = 0.5):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.chunks: Dict[str, Chunk] = {}
        self.links: Dict[str, List[SymbolicLink]] = {}  # target_id -> [links]
        self.chunk_ids: List[str] = []

        # Hyper-symbolic configuration
        self.use_mlp = use_mlp
        self.hyper_depth = hyper_depth
        self.lossy_threshold = lossy_threshold  # Links below this confidence are lossy

        # Link patterns for generating symbolic language
        self.link_patterns = {
            'analogy': '→',        # A is like B
            'contrast': '↔',       # A is not B
            'extension': '⇒',      # A leads to B
            'contradiction': '≠',  # A contradicts B
            'context': '≈',        # A is context for B
            'resolution': '✓',     # A resolves B
            'metaphor': '%%',      # A is metaphor for B
            'symbol': '∞',         # A symbolizes B
            'meta_analogy': '≡',   # A ≡ B (meta-level analogy)
            'meta_contrast': '≢',  # A ≢ B (meta-level contrast)
        }

        # Schema dimensions that can generate links
        self.schema_dimensions = {
            'geometry': ['linear', 'recursive', 'spiral', 'bifurcating'],
            'coherence': ['tight', 'loose', 'fragmented', 'emergent'],
            'texture': ['dense', 'sparse', 'lyrical', 'technical'],
            'terrain': ['conceptual', 'emotional', 'procedural', 'speculative'],
        }

        # MLP for link generation
        self.mlp = HyperSymbolicMLP() if use_mlp else None

    def add_chunk(self, text: str, embedding: List[float], tags: Dict[str, Any] = None) -> Chunk:
        """Add a chunk with its links to the index."""
        chunk_id = hashlib.sha256(text.encode()).hexdigest()[:12]

        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            embedding=embedding,
            schema_tags=tags or {}
        )

        self.chunks[chunk_id] = chunk
        self.chunk_ids.append(chunk_id)

        return chunk

    def generate_links(self, threshold: float = 0.3,
                       use_mlp: bool = None) -> List[SymbolicLink]:
        """
        Generate symbolic links between chunks based on schema similarity
        and/or MLP prediction.

        When schema tags match or complement, we create a link.
        The link value represents the strength of relationship.

        This is where the "language" is built: combinations of symbols
        create meaning through traversal.

        -- HYPER-SYMBOLIC --

        With MLP enabled, links are learned from embeddings + schema tags,
        producing probabilistic (lossy) relationships with confidence scores.
        """
        if use_mlp is None:
            use_mlp = self.use_mlp

        links = []

        for i, chunk1 in enumerate(self.chunks.values()):
            for chunk2 in list(self.chunks.values())[i+1:]:
                # Generate input vector for MLP
                input_vec = self.mlp.generate_input(
                    chunk1.embedding, chunk2.embedding,
                    chunk1.schema_tags, chunk2.schema_tags
                ) if self.mlp else []

                # Generate links via MLP or schema-based
                if use_mlp and self.mlp:
                    mlp_links = self.mlp.generate_links(
                        chunk1.embedding, chunk2.embedding,
                        chunk1.schema_tags, chunk2.schema_tags,
                        max_depth=self.hyper_depth
                    )

                    for link in mlp_links:
                        # Set chunk IDs
                        link.source_chunk_id = chunk1.chunk_id
                        link.target_chunk_id = chunk2.chunk_id

                        # Determine if lossy based on confidence
                        link.lossy = link.link_confidence < self.lossy_threshold

                        links.append(link)

                        # Store in both directions
                        if chunk2.chunk_id not in self.links:
                            self.links[chunk2.chunk_id] = []
                        self.links[chunk2.chunk_id].append(link)
                        chunk1.links.append(link)

                        # Add reverse link
                        reverse_link = SymbolicLink(
                            source_chunk_id=chunk2.chunk_id,
                            target_chunk_id=chunk1.chunk_id,
                            link_type=f"rev_{link.link_type}",
                            link_value=link.link_value,
                            link_symbols=link.link_symbols,
                            link_confidence=link.link_confidence,
                            lossy=link.lossy,
                            hyper_depth=link.hyper_depth
                        )
                        if chunk1.chunk_id not in self.links:
                            self.links[chunk1.chunk_id] = []
                        self.links[chunk1.chunk_id].append(reverse_link)
                        chunk2.links.append(reverse_link)

                else:
                    # Schema-based link generation (original logic)
                    overlap = 0
                    total = 0

                    for dim, tags1 in chunk1.schema_tags.items():
                        if dim in chunk2.schema_tags:
                            tags2 = chunk2.schema_tags[dim]
                            total += 1
                            if tags1 == tags2:
                                overlap += 1
                            elif self._are_complementary(tags1, tags2):
                                overlap += 0.5

                    if total > 0:
                        score = overlap / total
                        if score >= threshold:
                            link_type, link_symbols = self._determine_link_type(
                                chunk1.schema_tags, chunk2.schema_tags, score
                            )

                            link = SymbolicLink(
                                source_chunk_id=chunk1.chunk_id,
                                target_chunk_id=chunk2.chunk_id,
                                link_type=link_type,
                                link_value=score,
                                link_symbols=link_symbols,
                                link_confidence=score,  # Score as confidence
                                lossy=False,  # Schema-based is not lossy
                                hyper_depth=1
                            )
                            links.append(link)

                            # Store in both directions
                            if chunk2.chunk_id not in self.links:
                                self.links[chunk2.chunk_id] = []
                            self.links[chunk2.chunk_id].append(link)
                            chunk1.links.append(link)

                            reverse_link = SymbolicLink(
                                source_chunk_id=chunk2.chunk_id,
                                target_chunk_id=chunk1.chunk_id,
                                link_type=f"rev_{link_type}",
                                link_value=score,
                                link_symbols=link_symbols,
                                link_confidence=score,
                                lossy=False,
                                hyper_depth=1
                            )
                            if chunk1.chunk_id not in self.links:
                                self.links[chunk1.chunk_id] = []
                            self.links[chunk1.chunk_id].append(reverse_link)
                            chunk2.links.append(reverse_link)

        return links

    def _are_complementary(self, tag1: str, tag2: str) -> bool:
        """Check if two schema tags are complementary (can form links)."""
        complementary_pairs = [
            ('linear', 'recursive'),
            ('recursive', 'linear'),
            ('tight', 'loose'),
            ('loose', 'tight'),
            ('conceptual', 'procedural'),
            ('procedural', 'conceptual'),
        ]
        return (tag1, tag2) in complementary_pairs

    def _determine_link_type(self, tags1: Dict, tags2: Dict, score: float) -> Tuple[str, str]:
        """Determine link type and symbols based on schema relationship."""
        # Check for specific relationships
        if 'linear' in tags1.get('geometry', '') and 'recursive' in tags2.get('geometry', ''):
            return 'extension', '⇒'
        if tags1.get('texture') == tags2.get('texture'):
            return 'analogy', '→'
        if tags1.get('terrain') != tags2.get('terrain'):
            return 'contrast', '↔'

        # Default based on score
        if score > 0.8:
            return 'analogy', '→'
        elif score > 0.6:
            return 'context', '≈'
        else:
            return 'contrast', '↔'

    def find_path(self, start_id: str, end_id: str, max_depth: int = 3,
                  min_confidence: float = 0.0) -> Optional[List[SymbolicLink]]:
        """Find a path between two chunks through symbolic links."""
        if start_id not in self.chunks:
            return None

        chunk = self.chunks[start_id]
        paths = chunk.get_paths_to(end_id, max_depth, min_confidence)

        if not paths:
            return None

        # Return best path (most links, highest avg value)
        return max(paths, key=lambda p: len(p) * sum(l.link_value for l in p))

    def save(self, path: Optional[str] = None):
        """Save the symlink index to disk."""
        save_path = Path(path or self.index_path)
        save_path.mkdir(parents=True, exist_ok=True)

        data = {
            'chunks': {
                cid: {
                    'text': c.text,
                    'links': [
                        {
                            'target': l.target_chunk_id,
                            'type': l.link_type,
                            'value': l.link_value,
                            'symbols': l.link_symbols,
                            'confidence': l.link_confidence,
                            'lossy': l.lossy,
                            'hyper_depth': l.hyper_depth
                        }
                        for l in c.links
                    ],
                    'schema_tags': c.schema_tags
                }
                for cid, c in self.chunks.items()
            },
            'schema_dimensions': self.schema_dimensions,
            'hyper_symbolic': {
                'use_mlp': self.use_mlp,
                'hyper_depth': self.hyper_depth,
                'lossy_threshold': self.lossy_threshold
            },
            'timestamp': datetime.datetime.now().isoformat()
        }

        with open(save_path / 'symlink_index.json', 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> bool:
        """Load symlink index from disk."""
        load_path = Path(path or self.index_path)
        index_file = load_path / 'symlink_index.json'

        if not index_file.exists():
            return False

        with open(index_file) as f:
            data = json.load(f)

        # Load hyper-symbolic config
        hyper_config = data.get('hyper_symbolic', {})
        self.use_mlp = hyper_config.get('use_mlp', self.use_mlp)
        self.hyper_depth = hyper_config.get('hyper_depth', self.hyper_depth)
        self.lossy_threshold = hyper_config.get('lossy_threshold', self.lossy_threshold)

        self.chunks = {
            cid: Chunk(
                chunk_id=cid,
                text=c['text'],
                embedding=[],  # Would need to store/reload embeddings
                schema_tags=c.get('schema_tags', {}),
                links=[
                    SymbolicLink(
                        source_chunk_id=cid,
                        target_chunk_id=l['target'],
                        link_type=l['type'],
                        link_value=l['value'],
                        link_symbols=l['symbols'],
                        link_confidence=l.get('confidence', 1.0),
                        lossy=l.get('lossy', False),
                        hyper_depth=l.get('hyper_depth', 1)
                    )
                    for l in c.get('links', [])
                ]
            )
            for cid, c in data['chunks'].items()
        }

        # Rebuild links dict
        self.links = {}
        for cid, chunk in self.chunks.items():
            for link in chunk.links:
                if link.target_chunk_id not in self.links:
                    self.links[link.target_chunk_id] = []
                self.links[link.target_chunk_id].append(link)

        self.chunk_ids = list(self.chunks.keys())
        return True

    def analyze_link_patterns(self) -> Dict[str, Any]:
        """
        Analyze link patterns in the index.

        Returns statistics about:
        - Link type distribution
        - Lossy vs non-lossy ratio
        - Hyper-depth distribution
        - Connection patterns
        """
        stats = {
            'total_links': 0,
            'link_types': {},
            'lossy_ratio': 0,
            'hyper_depths': {},
            'avg_confidence': 0,
            'chunk_links': {}
        }

        all_confidences = []
        total_links = 0

        for cid, chunk in self.chunks.items():
            stats['chunk_links'][cid] = len(chunk.links)
            for link in chunk.links:
                total_links += 1
                all_confidences.append(link.link_confidence)

                # Link type stats
                link_type = link.link_type
                if link_type not in stats['link_types']:
                    stats['link_types'][link_type] = 0
                stats['link_types'][link_type] += 1

                # Lossy stats
                if link.lossy:
                    stats['lossy_ratio'] += 1

                # Hyper-depth stats
                depth = link.hyper_depth
                if depth not in stats['hyper_depths']:
                    stats['hyper_depths'][depth] = 0
                stats['hyper_depths'][depth] += 1

        if total_links > 0:
            stats['lossy_ratio'] /= total_links
            stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
            stats['total_links'] = total_links // 2  # Divide by 2 for bidirectional

        return stats


# ─── Query through Symlink Paths ────────────────────────────────


def query_via_paths(
    index: SymlinkIndex,
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
    max_path_depth: int = 3,
    min_confidence: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Query using path traversal through symbolic links.

    Instead of: "find closest vector"
    Do: "find start node → traverse paths → check endpoints"

    This leverages the language built from links.

    -- HYPER-SYMBOLIC --

    Supports filtering by confidence (lossy links) and hyper-depth.
    """
    results = []

    # Find starting chunks (those with highest vector similarity to query)
    # (In reality, we'd use FAISS here)
    start_chunks = list(index.chunks.values())[:top_k * 2]  # Top 2k candidates

    for start in start_chunks:
        # For each start, find paths to other chunks
        for end in list(index.chunks.values()):
            if start.chunk_id == end.chunk_id:
                continue

            paths = start.get_paths_to(end.chunk_id, max_path_depth, min_confidence)

            for path in paths:
                # Calculate path score
                path_score = sum(p.link_value for p in path) / len(path) if path else 0

                results.append({
                    'end_chunk': end,
                    'path': path,
                    'path_score': path_score,
                    'total_links': len(path),
                    'min_confidence': min(p.link_confidence for p in path) if path else 0,
                    'has_lossy': any(p.lossy for p in path)
                })

    # Sort by path score
    results.sort(key=lambda r: r['path_score'], reverse=True)

    return results[:top_k]


def link_language_query(
    index: SymlinkIndex,
    query_link_symbols: str,
    max_results: int = 5,
    min_confidence: float = 0.0,
    include_lossy: bool = True
) -> List[Dict[str, Any]]:
    """
    Query using link symbols as the language.

    Instead of words, you use symbols like:
    - '→' for analogy/extension
    - '↔' for contrast
    - '⇒' for resolution

    Example: "→recursive" finds things that extend/lead to recursive patterns.

    -- HYPER-SYMBOLIC --

    Supports filtering by confidence and inclusion of lossy links.
    """
    results = []

    for chunk in index.chunks.values():
        # Find chunks that have links with these symbols
        matching_links = [l for l in chunk.links if l.link_symbols in query_link_symbols]

        if matching_links:
            # Filter by confidence if specified
            if min_confidence > 0:
                matching_links = [l for l in matching_links if l.link_confidence >= min_confidence]

            # Filter lossy if requested
            if not include_lossy:
                matching_links = [l for l in matching_links if not l.lossy]

            if matching_links:
                # Calculate score based on link strength
                total_score = sum(l.link_value for l in matching_links)
                results.append({
                    'chunk': chunk,
                    'matching_links': matching_links,
                    'score': total_score / len(matching_links)
                })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:max_results]


# ─── Hyper-Symbolic Utilities ───────────────────────────────────


def analyze_link_patterns(index: SymlinkIndex) -> Dict[str, Any]:
    """
    Analyze link patterns in the index.

    Returns statistics about:
    - Link type distribution
    - Lossy vs non-lossy ratio
    - Hyper-depth distribution
    - Connection patterns
    """
    stats = {
        'total_links': 0,
        'link_types': {},
        'lossy_ratio': 0,
        'hyper_depths': {},
        'avg_confidence': 0,
        'chunk_links': {}
    }

    all_confidences = []
    total_links = 0

    for cid, chunk in index.chunks.items():
        stats['chunk_links'][cid] = len(chunk.links)
        for link in chunk.links:
            total_links += 1
            all_confidences.append(link.link_confidence)

            # Link type stats
            link_type = link.link_type
            if link_type not in stats['link_types']:
                stats['link_types'][link_type] = 0
            stats['link_types'][link_type] += 1

            # Lossy stats
            if link.lossy:
                stats['lossy_ratio'] += 1

            # Hyper-depth stats
            depth = link.hyper_depth
            if depth not in stats['hyper_depths']:
                stats['hyper_depths'][depth] = 0
            stats['hyper_depths'][depth] += 1

    if total_links > 0:
        stats['lossy_ratio'] /= total_links
        stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
        stats['total_links'] = total_links // 2  # Divide by 2 for bidirectional

    return stats


def print_link_summary(index: SymlinkIndex, limit: int = 10):
    """Print a summary of links in the index."""
    stats = analyze_link_patterns(index)

    print(f"Symlink Index Summary:")
    print(f"  Chunks: {len(index.chunks)}")
    print(f"  Total links (unique): {stats['total_links']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    print(f"  Lossy ratio: {stats['lossy_ratio']:.2%}")
    print()

    print("Link Types:")
    for lt, count in sorted(stats['link_types'].items(), key=lambda x: -x[1])[:limit]:
        print(f"  {lt}: {count}")
    print()

    print("Hyper-Depth Distribution:")
    for depth, count in sorted(stats['hyper_depths'].items(), key=lambda x: -x[1]):
        print(f"  Level {depth}: {count} links")
    print()

    if index.use_mlp:
        print("MLP Configuration:")
        print(f"  Use MLP: True")
        print(f"  Hyper-depth: {index.hyper_depth}")
        print(f"  Lossy threshold: {index.lossy_threshold}")
    else:
        print("MLP Configuration:")
        print(f"  Use MLP: False (schema-based only)")


# ─── Weighted Nearest Neighbor Synthesis ────────────────────────
# Words → Phrases → Sentences via symbolic context weighting


def nearest_neighbor_weighted(
    index: SymlinkIndex,
    query_embedding: List[float],
    query_context: Dict[str, Any] = None,
    k: int = 5,
    weight_by_symbolic_context: bool = True,
    weight_by_confidence: bool = True
) -> List[Dict[str, Any]]:
    """
    Weighted nearest neighbor retrieval using symbolic context.

    The weighting scheme:
    - Base: Vector similarity (cosine with query embedding)
    - Symbolic context: Boost by link type match with query context
    - Confidence: Weight by link confidence scores

    Args:
        index: The symlink index
        query_embedding: Query vector
        query_context: Dict with 'symbols' (link symbols) and 'link_types'
        k: Number of neighbors
        weight_by_symbolic_context: Boost by symbolic match
        weight_by_confidence: Boost by link confidence

    Returns:
        List of {chunk, score, neighbors} sorted by weighted score
    """
    results = []

    # Slicing MLP to determine which chunks are worth slicing
    slicing_mlp = SlicingMLP()

    for chunk in index.chunks.values():
        # Slicing MLP prediction
        should_slice, slice_importance = slicing_mlp.predict_slice(
            query_embedding, chunk.embedding, chunk.schema_tags
        )

        # Skip chunks not predicted to be sliceable (unless no results)
        if not should_slice and slice_importance < 0.7:
            continue

        # Base score: vector similarity (cosine approximation)
        # Simplified: dot product of embeddings
        base_score = 0.0
        if len(query_embedding) == len(chunk.embedding):
            base_score = sum(a * b for a, b in zip(query_embedding, chunk.embedding))

        # Symbolic context boost
        context_boost = 1.0
        if query_context and weight_by_symbolic_context:
            symbols = query_context.get('symbols', [])
            link_types = query_context.get('link_types', [])

            if symbols or link_types:
                matching_links = []
                for link in chunk.links:
                    symbol_match = symbols and link.link_symbols in symbols
                    type_match = link_types and link.link_type in link_types
                    if symbol_match or type_match:
                        matching_links.append(link)

                if matching_links:
                    # Weight by proportion of matching links and their confidence
                    avg_conf = sum(l.link_confidence for l in matching_links) / len(matching_links)
                    context_boost = 1.0 + (len(matching_links) / len(chunk.links)) * avg_conf

        # Confidence boost
        confidence_boost = 1.0
        if weight_by_confidence and chunk.links:
            avg_conf = sum(l.link_confidence for l in chunk.links) / len(chunk.links)
            confidence_boost = avg_conf

        # Combined weighted score (includes slice importance)
        final_score = base_score * context_boost * confidence_boost * slice_importance

        results.append({
            'chunk': chunk,
            'score': final_score,
            'base_score': base_score,
            'context_boost': context_boost,
            'confidence_boost': confidence_boost,
            'slice_importance': slice_importance,
            'neighbors': len(chunk.links)
        })

    # Fallback: if no results from slicing MLP, include all chunks
    if not results:
        for chunk in index.chunks.values():
            base_score = 0.0
            if len(query_embedding) == len(chunk.embedding):
                base_score = sum(a * b for a, b in zip(query_embedding, chunk.embedding))
            results.append({
                'chunk': chunk,
                'score': base_score,
                'base_score': base_score,
                'context_boost': 1.0,
                'confidence_boost': 1.0,
                'slice_importance': 0.5,
                'neighbors': len(chunk.links)
            })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:k]


def slice_and_splice(
    index: SymlinkIndex,
    source_ids: List[str],
    link_symbols: str,
    context_depth: int = 2,
    use_splicing_mlp: bool = True
) -> List[str]:
    """
    Slice words/phrases from sources and splice into new sequences
    via weighted symbolic context.

    Uses two MLPs:
    - SlicingMLP: Determines which chunks to extract (slicing)
    - SplicingMLP: Determines how to combine extracted chunks (splicing)

    Args:
        index: The symlink index
        source_ids: Source chunk IDs to slice from
        link_symbols: Target symbols (e.g., "→" for analogy links)
        context_depth: How many hops to include in context
        use_splicing_mlp: Whether to use SplicingMLP for optimal splice type

    Returns:
        List of spliced text segments
    """
    splices = []

    # Splicing MLP for optimal splice type prediction
    splicing_mlp = SplicingMLP() if use_splicing_mlp else None

    for source_id in source_ids:
        if source_id not in index.chunks:
            continue

        source_chunk = index.chunks[source_id]
        source_text = source_chunk.text

        # Find weighted links matching the symbols
        matching_links = []
        for link in source_chunk.links:
            if link.link_symbols in link_symbols:
                matching_links.append(link)

        if not matching_links:
            # Fall back to any links if none match
            matching_links = source_chunk.links[:3]

        # Collect neighbor context with weights
        neighbor_texts = []
        for link in matching_links:
            if link.target_chunk_id in index.chunks:
                neighbor = index.chunks[link.target_chunk_id]
                weight = link.link_confidence

                # Splicing MLP prediction for optimal combination
                splice_type, confidence, weights = None, None, None
                if splicing_mlp:
                    try:
                        splice_type, confidence, weights = splicing_mlp.predict_splice(
                            source_chunk.embedding, neighbor.embedding, link.link_type
                        )
                    except:
                        pass

                neighbor_texts.append({
                    'text': neighbor.text,
                    'weight': weight,
                    'link_type': link.link_type,
                    'splice_type': splice_type,
                    'confidence': confidence,
                    'weights': weights
                })

        # Sort neighbors by weight (highest first)
        neighbor_texts.sort(key=lambda x: x['weight'], reverse=True)

        # Create spliced output using SplicingMLP predictions
        if neighbor_texts:
            for i, neighbor_data in enumerate(neighbor_texts[:3]):  # Up to 3 splices
                primary = neighbor_data
                source_prefix = source_text[:60] if len(source_text) > 60 else source_text
                neighbor_prefix = primary['text'][:60] if len(primary['text']) > 60 else primary['text']

                # Determine splice connector based on predicted type or link type
                if primary['splice_type']:
                    splice_type = primary['splice_type']
                    if splice_type == 'analogical':
                        connector = " :: "
                    elif splice_type == 'contrastive':
                        connector = " but "
                    elif splice_type == 'sequential':
                        connector = " then "
                    elif splice_type == 'hierarchical':
                        connector = " :: "
                    else:  # concat, contextual
                        connector = " - "
                else:
                    connector = " "

                # Build spliced output
                if i == 0:
                    # Primary splice
                    spliced = f"{source_prefix}{connector}{neighbor_prefix}"
                else:
                    # Context chain
                    spliced = f"{source_prefix[:40]}... {connector} {neighbor_prefix[:40]}..."

                splices.append(spliced)

            # Add context chain if depth > 1
            if context_depth > 1 and neighbor_texts:
                first_neighbor_id = matching_links[0].target_chunk_id
                if first_neighbor_id in index.chunks:
                    neighbor_chunk = index.chunks[first_neighbor_id]
                    for next_link in neighbor_chunk.links[:2]:
                        if next_link.target_chunk_id in index.chunks:
                            next_text = index.chunks[next_link.target_chunk_id].text
                            spliced_chain = f"{source_text[:30]}... → {neighbor_texts[0]['text'][:30]}... → {next_text[:30]}..."
                            splices.append(spliced_chain)

        else:
            # No symbolic links, return source
            splices.append(source_text)

    return splices


def hierarchical_synthesis(
    index: SymlinkIndex,
    query: str,
    query_embedding: List[float],
    level: str = "phrase",  # "word", "phrase", "sentence"
    k: int = 3,
    max_depth: int = 2
) -> str:
    """
    Hierarchical synthesis: words → phrases → sentences.

    Uses weighted nearest neighbors and symbolic path traversal
    to construct output at the requested level.

    Args:
        index: The symlink index
        query: User query
        query_embedding: Query vector
        level: Output level ("word", "phrase", "sentence")
        k: Number of neighbors per level
        max_depth: Maximum path depth for traversal

    Returns:
        Synthesized text at the requested level
    """
    # Get weighted neighbors
    neighbors = nearest_neighbor_weighted(index, query_embedding, k=k)

    if not neighbors:
        return f"[No matches for query: {query}]"

    # Level-specific synthesis
    if level == "word":
        # Extract key words from highest-scoring neighbor
        chunk = neighbors[0]['chunk']
        words = chunk.text.split()[:10]  # First 10 words as "word" output
        return " ".join(words)

    elif level == "phrase":
        # Splice phrases from multiple neighbors
        phrases = []
        for neighbor in neighbors[:2]:  # Use top 2
            chunk = neighbor['chunk']
            # Get first meaningful phrase
            phrases.append(chunk.text[:100])

        return " | ".join(phrases)

    else:  # sentence
        # Build full sentence from path traversal
        paths_found = []
        start_chunk = neighbors[0]['chunk']

        for end_chunk in neighbors[1:]:
            paths = start_chunk.get_paths_to(end_chunk.chunk_id, max_depth=max_depth)
            if paths:
                paths_found.extend(paths[:2])  # Up to 2 paths per pair

        if paths_found:
            # Best path = highest confidence, longest
            best_path = max(paths_found, key=lambda p: len(p) * sum(l.link_confidence for l in p))

            # Build sentence from path
            sentence_parts = [start_chunk.text]
            for link in best_path:
                if link.target_chunk_id in index.chunks:
                    sentence_parts.append(index.chunks[link.target_chunk_id].text)

            return " ".join(sentence_parts[:5])  # Limit to 5 chunks

        else:
            # No paths, concatenate neighbors
            return " ".join(n['chunk'].text[:100] for n in neighbors[:3])


def context_weighted_path(
    index: SymlinkIndex,
    start_id: str,
    end_id: str,
    context_symbols: str,
    max_depth: int = 3
) -> Optional[List[SymbolicLink]]:
    """
    Find path with context-weighted links.

    Weights paths by:
    - Number of matching context symbols
    - Average link confidence
    - Path length (shorter preferred)

    Args:
        index: The symlink index
        start_id: Starting chunk
        end_id: Target chunk
        context_symbols: Required symbols for path
        max_depth: Maximum path length

    Returns:
        Weighted path or None
    """
    if start_id not in index.chunks:
        return None

    chunk = index.chunks[start_id]
    paths = chunk.get_paths_to(end_id, max_depth)

    if not paths:
        return None

    # Score paths by context match and confidence
    def path_score(path: List[SymbolicLink]) -> float:
        if not path:
            return 0.0

        # Symbol match score
        symbol_matches = sum(1 for l in path if l.link_symbols in context_symbols)
        symbol_score = symbol_matches / len(path)

        # Confidence score
        avg_conf = sum(l.link_confidence for l in path) / len(path)

        # Length penalty (shorter is better)
        length_score = 1.0 / len(path)

        return symbol_score * 0.4 + avg_conf * 0.4 + length_score * 0.2

    # Return best weighted path
    best_path = max(paths, key=path_score)
    return best_path


def print_synthesis_demo(index: SymlinkIndex, query: str, query_emb: List[float]):
    """Print synthesis demo for a query."""
    print(f"=== Hierarchical Synthesis Demo ===")
    print(f"Query: {query}")
    print()

    print("1. Nearest Neighbors (weighted):")
    neighbors = nearest_neighbor_weighted(index, query_emb, k=3)
    for n in neighbors:
        print(f"   {n['chunk'].chunk_id[:8]}: score={n['score']:.3f}, neighbors={n['neighbors']}")

    print()
    print("2. Slice and Splice (phrase level):")
    if index.chunks:
        source_ids = list(index.chunks.keys())[:2]
        splices = slice_and_splice(index, source_ids, "→", context_depth=2)
        for s in splices[:2]:
            print(f"   {s[:80]}...")

    print()
    print("3. Hierarchical Synthesis:")
    for level in ["word", "phrase", "sentence"]:
        result = hierarchical_synthesis(index, query, query_emb, level=level, k=3)
        print(f"   {level.capitalize()}: {result[:100]}")
