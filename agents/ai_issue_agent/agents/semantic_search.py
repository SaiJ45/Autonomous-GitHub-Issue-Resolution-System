import numpy as np
import re

try:
    from .embedding_utils import get_embeddings, expand_query
    from .chunking_utils import chunk_file
except ImportError:
    from embedding_utils import get_embeddings, expand_query
    from chunking_utils import chunk_file

# ------------------ KEYWORD MATCHING ------------------

def calculate_keyword_overlap(issue_text: str, file_text: str) -> float:
    """Calculate normalized token intersection over total issue tokens."""
    # Build stopword list inline for simplicity
    stopwords = {"in", "on", "the", "a", "an", "is", "for", "to", "and", "or", "of", "with", "this", "that", "it"}
    
    # Tokenize and normalize
    issue_tokens = set(re.findall(r"\w+", issue_text.lower())) - stopwords
    if not issue_tokens:
        return 0.0
        
    file_tokens = set(re.findall(r"\w+", file_text.lower()))
    
    intersection = issue_tokens.intersection(file_tokens)
    return len(intersection) / len(issue_tokens)


# ------------------ LANGUAGE DETECTION ------------------

def get_language_boost(issue_text: str, file_path: str) -> float:
    """Soft boost for matching language context rather than hard filtering."""
    issue = issue_text.lower()
    path = file_path.lower()
    
    boost = 0.0
    if any(word in issue for word in ["python", "function", "error", "divide", "bug"]) and path.endswith(".py"):
        boost = 0.05
    elif ("javascript" in issue or "js" in issue) and path.endswith(".js"):
        boost = 0.05
    elif "html" in issue and path.endswith(".html"):
        boost = 0.05
    elif "css" in issue and path.endswith(".css"):
        boost = 0.05
        
    return boost


# ------------------ EMBEDDINGS ------------------

def build_embeddings(files):
    """
    Deprecated: Preserved for API backward compatibility.
    Computes global file embeddings. Semantic search now uses chunk-level embeddings dynamically.
    """
    texts = [f["content"][:2000] for f in files]
    return get_embeddings(texts)


# ------------------ CORE SEARCH ------------------

def find_relevant_files(issue_text: str, files: list[dict], embeddings=None, top_k: int = 5) -> list[dict]:
    """
    Industry-grade semantic search using chunking, dense embeddings, and robust scoring.
    """
    if not files:
        return []

    # 1. Expand query and compute normalized query embedding
    expanded_issue = expand_query(issue_text)
    query_emb = get_embeddings([expanded_issue], is_query=True)[0]
    
    # 2. Chunk all files
    all_chunks = []
    chunk_to_file_idx = []
    
    for file_idx, file in enumerate(files):
        path = file.get("path", "")
        content = file.get("content", "")
        file_chunks = chunk_file(path, content, chunk_size=300, overlap=50)
        
        for chunk in file_chunks:
            all_chunks.append(chunk)
            chunk_to_file_idx.append(file_idx)
            
    if not all_chunks:
        return []

    # 3. Compute chunk embeddings
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    chunk_embs = get_embeddings(chunk_texts)
    
    # 4. Compute similarity for all chunks efficiently (inner product since normalized)
    # Using np.dot matrix multiplication for fast computation
    semantic_scores = np.dot(chunk_embs, query_emb)
    
    # Track the best chunk score for each file
    best_file_scores = {}
    
    for i, chunk in enumerate(all_chunks):
        file_idx = chunk_to_file_idx[i]
        file_dict = files[file_idx]
        path = file_dict["path"].replace("\\", "/").lower()
        chunk_text = chunk["chunk_text"]
        
        # Component 1: Semantic (normalized 0-1)
        sem_score = max(0.0, float(semantic_scores[i]))
        
        # Component 2: Keyword Overlap (normalized 0-1)
        kw_score = calculate_keyword_overlap(expanded_issue, chunk_text)
        
        # Component 3: Path Relevance (normalized roughly 0-1)
        path_score = calculate_keyword_overlap(issue_text, path)
        
        # Calculate base score with weighted components
        base_score = (0.8 * sem_score) + (0.15 * kw_score) + (0.05 * path_score)
        
        # Apply soft language boost
        lang_boost = get_language_boost(issue_text, path)
        final_score = base_score + lang_boost
        
        # Keep maximum scored chunk per file
        if file_idx not in best_file_scores or final_score > best_file_scores[file_idx]["score"]:
            best_file_scores[file_idx] = {
                "score": final_score,
                "sem_score": sem_score,
                "kw_score": kw_score,
                "path_score": path_score,
                "chunk_text": chunk_text,
                "chunk_idx": chunk["chunk_idx"]
            }

    # 5. Extract top results and attach explainability metadata
    ranked_indices = sorted(best_file_scores.keys(), key=lambda idx: best_file_scores[idx]["score"], reverse=True)
    
    top_files = []
    for idx in ranked_indices[:top_k]:
        file_dict = files[idx].copy()  # Use copy to avoid mutating original dictionary if shared
        meta = best_file_scores[idx]
        
        file_dict["search_score"] = float(meta["score"])
        file_dict["search_reason"] = (
            f"Score: {meta['score']:.3f} (Sem: {meta['sem_score']:.3f}, Kw: {meta['kw_score']:.3f}, Path: {meta['path_score']:.3f}). "
            f"Best chunk idx: {meta['chunk_idx']}."
        )
        file_dict["top_match_chunk"] = meta["chunk_text"]
        
        top_files.append(file_dict)
        
    return top_files
