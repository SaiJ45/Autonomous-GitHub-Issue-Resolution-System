import hashlib
import numpy as np
import re

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("[WARN] sentence-transformers / faiss not installed. Semantic search will degrade gracefully.")

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model_instance = None
_embedding_cache = {}

from functools import lru_cache

@lru_cache(maxsize=1)
def load_embedding_model():
    """Lazy load and strongly cache the embedding model."""
    if not EMBEDDING_AVAILABLE:
        return None
    print(f"[INFO] Loading embedding model ({MODEL_NAME})...")
    return SentenceTransformer(MODEL_NAME)

def get_embedding_model():
    return load_embedding_model()

def get_embeddings(texts: list[str], is_query=False) -> np.ndarray:
    """
    Get normalized embeddings for a list of texts.
    Caches results based on string hashes to optimize performance.
    """
    model = get_embedding_model()
    if not model:
        # Fallback dummy embeddings if model unavailable
        return np.zeros((len(texts), 768), dtype="float32")

    if is_query:
        # BGE model prefix for queries
        texts = [QUERY_PREFIX + t for t in texts]

    embeddings = []
    texts_to_compute = []
    indices_to_compute = []

    for i, text in enumerate(texts):
        # Use simple MD5 hash for cache key
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in _embedding_cache:
            embeddings.append(_embedding_cache[text_hash])
        else:
            embeddings.append(None)
            texts_to_compute.append(text)
            indices_to_compute.append((i, text_hash))

    if texts_to_compute:
        computed = model.encode(texts_to_compute, show_progress_bar=False, batch_size=32)
        computed = np.array(computed, dtype="float32")
        # Normalize
        norms = np.linalg.norm(computed, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1e-10
        computed = computed / norms
        
        for (orig_idx, test_hash), emb in zip(indices_to_compute, computed):
            _embedding_cache[test_hash] = emb
            embeddings[orig_idx] = emb

    return np.array(embeddings, dtype="float32")

def expand_query(issue_text: str) -> str:
    """Expand common issues with synonyms for better semantic matching."""
    synonyms = {
        "bug": "bug error failure crash unexpected incorrect flaw",
        "fix": "fix resolve patch correct address repair",
        "add": "add implement create support new feature",
        "error": "error exception traceback fault crash",
        "update": "update modify change refactor improve",
    }
    
    expanded = issue_text
    words = set(re.findall(r"\w+", issue_text.lower()))
    
    for key, syns in synonyms.items():
        if key in words:
            expanded += " " + syns
            
    return expanded
