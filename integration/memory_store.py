import os
import json
import logging
import shutil
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logging.warning("sentence-transformers not available, memory_store will use exact matching.")

MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'memory_store.json')
MEMORY_BACKUP_DIR = os.path.join(os.path.dirname(__file__), '.memory_backups')
MAX_MEMORY_SIZE = 50

# Maintain a small LRU cache for the embedding model if we need it
_MODEL = None

def _ensure_backup_dir():
    """Ensure backup directory exists."""
    try:
        os.makedirs(MEMORY_BACKUP_DIR, exist_ok=True)
    except Exception as e:
        logging.warning(f"Failed to create backup directory: {e}")


def _backup_corrupted_file(error_reason: str) -> str:
    """
    Backup the corrupted memory_store.json file with timestamp.
    
    Args:
        error_reason: Description of why the file is corrupted.
    
    Returns:
        Path to the backup file.
    """
    try:
        _ensure_backup_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            MEMORY_BACKUP_DIR,
            f"memory_store_corrupted_{timestamp}.json"
        )
        
        if os.path.exists(MEMORY_FILE):
            shutil.copy2(MEMORY_FILE, backup_path)
            logging.warning(
                f"Backed up corrupted memory_store.json to {backup_path} "
                f"(reason: {error_reason})"
            )
            return backup_path
    except Exception as e:
        logging.error(f"Failed to backup corrupted file: {e}")
    
    return None


def _json_safe(value: Any) -> Any:
    """Recursively coerce values into JSON-safe data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            pass
    return str(value)


def _normalize_memory_record(record: Any) -> Dict[str, Any] | None:
    """Normalize a memory record into a JSON-safe dict."""
    if not isinstance(record, dict):
        return None

    normalized = _json_safe(record)
    if not isinstance(normalized, dict):
        return None

    issue_text = str(normalized.get("issue", "")).strip()
    if not issue_text:
        return None

    normalized["issue"] = issue_text
    normalized.setdefault("saved_at", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    return normalized


def _write_memory_file(memory: List[Dict]) -> None:
    """Persist memory atomically to disk."""
    temp_path = MEMORY_FILE + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp_path, MEMORY_FILE)


def _reset_memory_store() -> None:
    """Reset the memory store to a known-good empty list."""
    try:
        _write_memory_file([])
    except Exception as e:
        logging.error(f"Failed to reset memory_store.json: {e}")


def _get_model():
    global _MODEL
    if EMBEDDING_AVAILABLE and _MODEL is None:
        try:
            _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception:
            logging.error("Failed to load embedding model.")
    return _MODEL

def _load_memory() -> List[Dict]:
    """
    Load memory from memory_store.json with safe fallback.
    
    On corruption:
    1. Backup the corrupted file
    2. Log the error
    3. Return empty list to continue operation
    
    Returns:
        List of memory dicts, or [] if load fails.
    """
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"memory_store root is not a list, got {type(data).__name__}")
        cleaned: List[Dict] = []
        dropped = 0
        for item in data:
            normalized = _normalize_memory_record(item)
            if normalized is None:
                dropped += 1
                continue
            cleaned.append(normalized)
        if dropped:
            logging.warning(f"Dropped {dropped} invalid memory record(s) while loading memory_store.json.")
        return cleaned
    except json.JSONDecodeError as e:
        error_msg = f"JSONDecodeError: {e.msg} at line {e.lineno}"
        logging.error(f"Corrupted memory_store.json ({error_msg})")
        _backup_corrupted_file(error_msg)
        _reset_memory_store()
        return []
    except ValueError as e:
        error_msg = str(e)
        logging.error(f"Invalid memory_store structure: {error_msg}")
        _backup_corrupted_file(error_msg)
        _reset_memory_store()
        return []
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logging.error(f"Error reading memory_store: {error_msg}")
        _backup_corrupted_file(error_msg)
        _reset_memory_store()
        return []

def _save_memory(memory: List[Dict]):
    """
    Save memory to memory_store.json with atomic write.
    
    Uses a temporary file to avoid partial writes on crash.
    
    Args:
        memory: List of memory dicts to save.
    """
    try:
        _ensure_backup_dir()
        serialized_memory: List[Dict] = []
        for item in memory:
            normalized = _normalize_memory_record(item)
            if normalized is None:
                logging.warning("Skipping invalid memory record during save.")
                continue
            serialized_memory.append(normalized)
        _write_memory_file(serialized_memory)
    except Exception as e:
        logging.error(f"Error writing memory_store: {e}")
        # Clean up temp file if it exists
        temp_path = MEMORY_FILE + ".tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def search_memory(issue_description: str, top_k: int = 3) -> List[Dict]:
    """
    Search past resolved issues using embedding similarity.
    Fallback to word overlap if embeddings not available.
    """
    memory = _load_memory()
    if not memory:
        return []

    model = _get_model()
    if EMBEDDING_AVAILABLE and model is not None:
        # Re-compute embeddings dynamically for simplicity, or we could cache them
        texts = [m.get("issue", "") for m in memory]
        try:
            query_emb = model.encode([issue_description])
            doc_embs = model.encode(texts)
            sims = cosine_similarity(query_emb, doc_embs)[0]
            
            # Sort by similarity descending
            scored = list(zip(sims, memory))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Filter threshold to avoid completely unrelated memories
            return [mem for score, mem in scored[:top_k] if score > 0.4]
        except Exception as e:
            logging.error(f"Embedding search failed: {e}")
            pass

    # Fallback to simple keyword overlap
    query_words = set(issue_description.lower().split())
    scored = []
    for m in memory:
        mem_words = set(m.get("issue", "").lower().split())
        overlap = len(query_words.intersection(mem_words))
        scored.append((overlap, m))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [mem for score, mem in scored[:top_k] if score > 0]

def save_to_memory(record: Dict):
    """
    Save a resolved issue schema to long-term memory.
    Expected record:
    {
        "issue": str,
        "solution": str,
        "tests": [str],
        "patterns": [str]
    }
    """
    normalized_record = _normalize_memory_record(record)
    if normalized_record is None:
        logging.warning("Skipping memory save for invalid record.")
        return

    memory = _load_memory()
    
    # Deduplicate (if very similar issue is already there, overwrite it)
    model = _get_model()
    if EMBEDDING_AVAILABLE and model is not None and memory:
        texts = [m.get("issue", "") for m in memory]
        try:
            q_emb = model.encode([normalized_record.get("issue", "")])
            d_embs = model.encode(texts)
            sims = cosine_similarity(q_emb, d_embs)[0]
            max_idx = np.argmax(sims)
            if sims[max_idx] > 0.95:  # Almost identical issue
                memory[max_idx] = normalized_record
                _save_memory(memory)
                return
        except Exception:
            pass

    memory.append(normalized_record)
    
    # Prune oldest if MAX limit exceeded
    if len(memory) > MAX_MEMORY_SIZE:
        memory = memory[-MAX_MEMORY_SIZE:]
        
    _save_memory(memory)
