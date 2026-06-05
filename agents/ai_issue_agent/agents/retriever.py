"""
agents/retriever.py

Hybrid file retrieval system:
  1. AST-based function-level chunking (structural hints)
  2. FAISS + sentence-transformers semantic search
  3. Keyword-based scoring
  4. Combined ranking -> top-5 candidate files
"""

import os
import ast
import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("[WARN] sentence-transformers / faiss not installed. Falling back to keyword-only mode.")

from functools import lru_cache

try:
    from ..config import CLONE_PATH
except ImportError:
    from config import CLONE_PATH

@lru_cache(maxsize=1)
def load_faiss_index(index_path: str):
    import faiss
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None

@lru_cache(maxsize=1)
def load_chunks_cache(chunks_path: str):
    import json
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Multi-language support: maps extension -> chunking strategy
# ONLY source code files — never docs, configs, artifacts, or logs
LANGUAGE_EXTENSIONS = {
    ".py": "ast",
    ".ts": "regex", ".tsx": "regex", ".js": "regex", ".jsx": "regex",
    ".java": "line", ".go": "line", ".rs": "line", ".rb": "line",
    ".cpp": "line", ".c": "line", ".cs": "line",
}
ALLOWED_EXTENSIONS = tuple(LANGUAGE_EXTENSIONS.keys())
STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "not", "with", "this", "that", "be", "are", "was", "were",
    "has", "have", "had", "do", "does", "did", "fix", "bug", "issue",
    "error", "problem", "should", "when", "how", "why", "what", "where",
}

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Language detection mappings
LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", 
    ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".go": "golang", ".rs": "rust", 
    ".rb": "ruby", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
}

ISSUE_TYPE_LANGUAGES = {
    "backend": {".py", ".java", ".ts", ".go", ".rb", ".cpp", ".c", ".cs"},
    "frontend": {".js", ".jsx", ".ts", ".tsx", ".html", ".css"},
    "config": {".json", ".yaml", ".yml", ".env", ".toml", ".ini"},
    "database": {".py", ".sql", ".java"},
    "api": {".py", ".ts", ".java", ".go", ".rb"},
}

# Down-rank file patterns (unless they match issue)
DOWNRANK_PATTERNS = {
    "dist/", "build/", "node_modules/", "__pycache__/", ".git/",
    "migrations/", "fixtures/", ".min.js", ".min.css", ".bundle.",
    "vendor/", "coverage/", "test_", "_test.", "spec_", ".spec.",
    "manage.py", "management/", "conftest", "setup.py", "wsgi.py", "asgi.py",
}

# Priority file patterns — boosted in scoring (Django / common web frameworks)
PRIORITY_BASENAMES = {
    "views.py", "models.py", "forms.py", "serializers.py", "urls.py",
    "admin.py", "signals.py", "utils.py", "helpers.py", "services.py",
}

# ---------------------------------------------------------------------------
# Language Detection & Filtering
# ---------------------------------------------------------------------------

def _detect_issue_type(issue_text: str) -> str:
    """Infer issue type (backend/frontend/config/database/api) from issue text."""
    text_lower = issue_text.lower()
    
    # Database-related keywords
    if any(kw in text_lower for kw in ["database", "query", "sql", "migration", "schema", "orm"]):
        return "database"
    
    # Frontend-related keywords
    if any(kw in text_lower for kw in ["ui", "button", "form", "render", "component", "css", "html", "react", "vue", "angular"]):
        return "frontend"
    
    # API-related keywords
    if any(kw in text_lower for kw in ["api", "endpoint", "request", "response", "http", "rest"]):
        return "api"
    
    # Config-related keywords
    if any(kw in text_lower for kw in ["config", "environment", "settings", ".env", "yaml", "json"]):
        return "config"
    
    # Default to backend
    return "backend"


def _should_downrank(file_path: str, issue_text: str) -> bool:
    """Check if file should be down-ranked based on patterns."""
    path_lower = file_path.lower()
    issue_lower = issue_text.lower()
    
    # Extract file basename and directory pattern
    basename = os.path.basename(path_lower)
    dirname = os.path.dirname(path_lower)
    
    # If file matches downrank patterns AND issue doesn't mention it specifically
    for pattern in DOWNRANK_PATTERNS:
        if pattern in path_lower:
            # Check if issue explicitly references this file/directory
            if basename.split(".")[0] not in issue_lower and pattern.strip("/") not in issue_lower:
                return True
    
    return False


def _score_language_relevance(file_ext: str, issue_type: str) -> float:
    """Score file language relevance (0.0 to 1.0)."""
    if issue_type not in ISSUE_TYPE_LANGUAGES:
        return 0.5
    
    target_exts = ISSUE_TYPE_LANGUAGES[issue_type]
    if file_ext in target_exts:
        return 1.0
    
    # Give slight credit to related languages
    if issue_type == "backend" and file_ext in {".py", ".ts", ".java"}:
        return 0.8
    if issue_type == "frontend" and file_ext in {".ts", ".tsx", ".js", ".jsx"}:
        return 0.9
    
    return 0.3


def _get_repo_languages(repo_path: str) -> dict[str, int]:
    """Count files by language to detect dominant languages."""
    lang_counts = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", "env", "node_modules")]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in LANGUAGE_BY_EXTENSION:
                lang = LANGUAGE_BY_EXTENSION[ext]
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
    return lang_counts

# ---------------------------------------------------------------------------
# AST Utilities
# ---------------------------------------------------------------------------

def extract_ast_chunks(file_path: str, source: str) -> list[dict]:
    """
    Parse a Python file with ast and return one chunk per function/method.
    Each chunk contains:
      - function_name
      - file_path
      - snippet (source lines of the function)
      - class_name (if inside a class, else None)
      - imports (top-level import names in the file)
    """
    chunks = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines()

    # Collect top-level imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []):
                imports.append(alias.name)
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

    def _extract_functions(node, class_name=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                _extract_functions(child, class_name=child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = child.lineno - 1
                end = child.end_lineno
                snippet = "\n".join(source_lines[start:end])
                chunks.append({
                    "function_name": child.name,
                    "class_name": class_name,
                    "file_path": file_path,
                    "snippet": snippet,
                    "imports": imports,
                    "language": LANGUAGE_BY_EXTENSION.get(os.path.splitext(file_path)[1].lower(), "unknown"),
                })
                # Recurse for nested functions
                _extract_functions(child, class_name=class_name)

    _extract_functions(tree)

    # If no functions found, treat whole file as one chunk
    if not chunks:
        chunks.append({
            "function_name": "__module__",
            "class_name": None,
            "file_path": file_path,
            "snippet": source[:3000],
            "imports": imports,
            "language": LANGUAGE_BY_EXTENSION.get(os.path.splitext(file_path)[1].lower(), "unknown"),
        })

    return chunks


def extract_file_structure(file_path: str, source: str) -> dict:
    """
    Return top-level structural info about a file (for display/logging).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"functions": [], "classes": [], "imports": []}

    functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes   = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports   = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

    return {"functions": functions, "classes": classes, "imports": imports}


# ---------------------------------------------------------------------------
# Non-Python Chunkers
# ---------------------------------------------------------------------------

def extract_regex_chunks(file_path: str, source: str) -> list[dict]:
    """Regex-based function extraction for JS/TS files."""
    chunks = []
    pattern = re.compile(
        r'(?:(?:export\s+)?(?:async\s+)?function\s+(\w+)|'
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+?)\s*=>)',
        re.MULTILINE,
    )
    lines = source.splitlines()
    matches = list(pattern.finditer(source))
    lang = LANGUAGE_BY_EXTENSION.get(os.path.splitext(file_path)[1].lower(), "unknown")
    for i, m in enumerate(matches):
        name = m.group(1) or m.group(2) or "anonymous"
        start_line = source[:m.start()].count("\n")
        end_line = source[:matches[i + 1].start()].count("\n") if i + 1 < len(matches) else len(lines)
        snippet = "\n".join(lines[start_line:min(start_line + 60, end_line)])
        chunks.append({
            "function_name": name,
            "class_name": None,
            "file_path": file_path,
            "snippet": snippet,
            "imports": [],
            "language": lang,
        })
    if not chunks:
        chunks.append({
            "function_name": "__module__",
            "class_name": None,
            "file_path": file_path,
            "snippet": source[:2000],
            "imports": [],
            "language": lang,
        })
    return chunks


def extract_line_chunks(file_path: str, source: str, window: int = 100) -> list[dict]:
    """Line-window chunking for Java/Go/Rust/Ruby and similar languages."""
    lines = source.splitlines()
    chunks = []
    lang = LANGUAGE_BY_EXTENSION.get(os.path.splitext(file_path)[1].lower(), "unknown")
    for i in range(0, max(1, len(lines)), window):
        snippet = "\n".join(lines[i:i + window])
        chunks.append({
            "function_name": f"lines_{i}_{i + window}",
            "class_name": None,
            "file_path": file_path,
            "snippet": snippet,
            "imports": [],
            "language": lang,
        })
    return chunks or [{"function_name": "__module__", "class_name": None,
                       "file_path": file_path, "snippet": source[:2000], "imports": [], 
                       "language": lang}]


def extract_generic_chunk(file_path: str, source: str) -> list[dict]:
    """Whole-file chunk for config/markup files."""
    lang = LANGUAGE_BY_EXTENSION.get(os.path.splitext(file_path)[1].lower(), "unknown")
    return [{
        "function_name": "__file__",
        "class_name": None,
        "file_path": file_path,
        "snippet": source[:2000],
        "imports": [],
        "language": lang,
    }]


def chunk_file(file_path: str, source: str) -> list[dict]:
    """Dispatch to the correct chunker based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    strategy = LANGUAGE_EXTENSIONS.get(ext, "generic")
    if strategy == "ast":
        return extract_ast_chunks(file_path, source)
    elif strategy == "regex":
        return extract_regex_chunks(file_path, source)
    elif strategy == "line":
        return extract_line_chunks(file_path, source)
    else:
        return extract_generic_chunk(file_path, source)


# ---------------------------------------------------------------------------
# Keyword Utilities
# ---------------------------------------------------------------------------

def extract_keywords(text: str) -> list[str]:
    """Lower-cased, stop-word filtered tokens from issue text."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    return [t.lower() for t in tokens if t.lower() not in STOPWORDS and len(t) > 2]


def keyword_score(chunk: dict, keywords: list[str]) -> float:
    """
    Score a chunk based on keyword overlap with:
      - function name
      - class name
      - file path (basename + dirs)
      - snippet content (limited to first 1000 chars)
    """
    score = 0.0
    fn   = (chunk.get("function_name") or "").lower()
    cn   = (chunk.get("class_name")    or "").lower()
    path = chunk["file_path"].replace("\\", "/").lower()
    code = chunk["snippet"][:1000].lower()

    for kw in keywords:
        if kw == fn:
            score += 4.0   # exact function match
        elif kw in fn:
            score += 3.0   
        if kw == cn:
            score += 4.0   # exact class match
        elif kw in cn:
            score += 2.0
        if kw in os.path.basename(path):
            score += 2.0
        if kw in path:
            score += 1.0
        if kw in code:
            score += 0.5

    return score


def path_keyword_score(path: str, keywords: list[str]) -> float:
    """Extra relevance score for strong path and filename matches."""
    if not isinstance(path, str):
        return 0.0

    normalized = path.replace("\\", "/").lower()
    basename = os.path.basename(normalized)
    stem, _ = os.path.splitext(basename)
    score = 0.0

    for kw in keywords:
        if kw == stem:
            score += 4.0
        elif kw in stem:
            score += 2.5
        elif kw in basename:
            score += 2.0
        elif kw in normalized:
            score += 1.0

    return score


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Build a FAISS index over function-level chunks from a cloned repository.
    Query with an issue string -> returns top-k ranked candidates.
    """

    def __init__(self, repo_path: str = CLONE_PATH):
        self.repo_path = repo_path
        self.chunks: list[dict] = []
        self.index = None
        self.model = None

        if EMBEDDING_AVAILABLE:
            from embedding_utils import get_embedding_model
            self.model = get_embedding_model()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _collect_chunks(self) -> list[dict]:
        chunks = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden dirs, __pycache__, .git, venv
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", "env", "node_modules")]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in LANGUAGE_EXTENSIONS:
                    continue
                
                # Exclude tests and generated artifacts by name
                fname_lower = fname.lower()
                if "test" in fname_lower:
                    continue
                if "qa_report" in fname_lower or "artifact" in fname_lower or "summary" in fname_lower:
                    continue
                
                full_path = os.path.join(root, fname)
                rel_path  = os.path.relpath(full_path, self.repo_path)
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        source = f.read()
                except Exception:
                    continue
                file_chunks = chunk_file(rel_path, source)
                chunks.extend(file_chunks)
        return chunks

    def build_index(self):
        """Parse all repo files, chunk by function/strategy, build/load FAISS index."""
        import json

        cache_dir = os.path.join(os.path.dirname(self.repo_path), ".ai_cache")
        os.makedirs(cache_dir, exist_ok=True)
        index_path  = os.path.join(cache_dir, "faiss.index")
        chunks_path = os.path.join(cache_dir, "chunks.json")

        # Cache invalidation: rebuild if any tracked file is newer than cache
        def _cache_is_valid() -> bool:
            if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
                return False
            cache_mtime = os.path.getmtime(index_path)
            for root, dirs, files in os.walk(self.repo_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", "env", "node_modules")]
                for fname in files:
                    if os.path.splitext(fname)[1].lower() in LANGUAGE_EXTENSIONS:
                        if os.path.getmtime(os.path.join(root, fname)) > cache_mtime:
                            return False
            return True

        if EMBEDDING_AVAILABLE and _cache_is_valid():
            try:
                print("[INDEX] Loading cached FAISS index from disk...")
                loaded_chunks = load_chunks_cache(chunks_path)
                loaded_index  = load_faiss_index(index_path)
                if loaded_chunks and loaded_index:
                    self.chunks = loaded_chunks
                    self.index  = loaded_index
                    print(f"   -> Loaded {len(self.chunks)} chunks [OK] via cache")
                    return
            except Exception as e:
                print(f"[WARN] Failed to load cache, rebuilding: {e}")

        print("[INDEX] Building hybrid retrieval index...")
        self.chunks = self._collect_chunks()

        if not self.chunks:
            print("[WARN] No chunks found in repository.")
            return

        print(f"   -> {len(self.chunks)} function chunks indexed")

        if EMBEDDING_AVAILABLE and self.model:
            texts = [
                f"{c['function_name']} {c.get('class_name', '')} {c['snippet'][:500]}"
                for c in self.chunks
            ]
            embeddings = self.model.encode(texts, show_progress_bar=False, batch_size=32)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)    # inner product on normalized = cosine
            self.index.add(embeddings)
            
            # Save cache
            try:
                faiss.write_index(self.index, index_path)
                with open(chunks_path, "w", encoding="utf-8") as f:
                    json.dump(self.chunks, f)
            except Exception as e:
                print(f"[WARN] Failed to save FAISS cache: {e}")
                
            print("   -> FAISS index built and cached [OK]")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, issue_text: str, top_k: int = 10) -> list[dict]:
        """
        Returns up to top_k candidate dicts with language-aware filtering.
        Each result includes: {path, function_name, snippet, combined_score, 
        semantic_score, keyword_score_val, structure, language}
        """
        if not self.chunks:
            print("[WARN] Index is empty -- call build_index() first.")
            return []

        keywords = extract_keywords(issue_text)
        issue_type = _detect_issue_type(issue_text)
        repo_languages = _get_repo_languages(self.repo_path)
        
        print(f"[RETRIEVAL] Issue type: {issue_type}, dominant repo languages: {repo_languages}")

        # ------ Semantic scores ------
        semantic_scores: dict[int, float] = {}

        if EMBEDDING_AVAILABLE and self.model and self.index is not None:
            query_with_prefix = QUERY_PREFIX + issue_text
            q_emb = self.model.encode([query_with_prefix], show_progress_bar=False)
            q_emb = np.array(q_emb, dtype="float32")
            faiss.normalize_L2(q_emb)

            n_search = min(len(self.chunks), max(top_k * 10, 50))
            distances, indices = self.index.search(q_emb, n_search)

            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:
                    semantic_scores[idx] = float(dist)

        # ------ Combine scores per chunk with language awareness ------
        chunk_scores: list[tuple[float, float, float, int]] = []  # (combined, sem, kw, idx)

        for i, chunk in enumerate(self.chunks):
            # Skip downranked files unless they're explicitly mentioned in issue
            if _should_downrank(chunk["file_path"], issue_text):
                continue
            
            sem  = semantic_scores.get(i, 0.0)
            kw   = keyword_score(chunk, keywords)
            path_kw = path_keyword_score(chunk["file_path"], keywords)
            fn_name = (chunk.get("function_name") or "").lower()
            fn_exact = 1.0 if any(kwd == fn_name for kwd in keywords) else 0.0
            
            # Language relevance scoring
            file_ext = os.path.splitext(chunk["file_path"])[1].lower()
            lang_relevance = _score_language_relevance(file_ext, issue_type)
            
            # Normalize scores
            kw_norm = min(kw / 20.0, 1.0)
            path_norm = min(path_kw / 10.0, 1.0)
            
            # Priority boost for core application files (views.py, models.py, etc.)
            basename = os.path.basename(chunk["file_path"]).lower()
            priority_boost = 0.15 if basename in PRIORITY_BASENAMES else 0.0
            
            # Language-aware weighted combination
            combined = (
                0.40 * sem +           # semantic similarity
                0.20 * kw_norm +       # keyword matching
                0.15 * path_norm +     # path/filename relevance
                0.10 * lang_relevance +  # language match for issue type
                0.05 * fn_exact +      # exact function name match
                0.10 * priority_boost  # core file boost
            )
            chunk_scores.append((combined, sem, kw + path_kw, i))

        chunk_scores.sort(key=lambda x: x[0], reverse=True)

        # ------ Deduplicate by file path (keep best chunk per file) ------
        seen_paths: dict[str, tuple] = {}
        for combined, sem, kw, idx in chunk_scores:
            path = self.chunks[idx]["file_path"]
            if path not in seen_paths or combined > seen_paths[path][0]:
                seen_paths[path] = (combined, sem, kw, idx)

        # Sort deduplicated paths by combined score
        ranked = sorted(seen_paths.values(), key=lambda x: x[0], reverse=True)

        results = []
        for combined, sem, kw, idx in ranked[:top_k]:
            chunk = self.chunks[idx]
            file_ext = os.path.splitext(chunk["file_path"])[1].lower()
            lang = LANGUAGE_BY_EXTENSION.get(file_ext, "unknown")
            
            # Read structure for display
            full_path = os.path.join(self.repo_path, chunk["file_path"])
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                structure = extract_file_structure(chunk["file_path"], source)
            except Exception:
                structure = {"functions": [], "classes": [], "imports": []}

            results.append({
                "path":            chunk["file_path"],
                "function_name":   chunk["function_name"],
                "snippet":         chunk["snippet"],
                "combined_score":  round(combined, 4),
                "semantic_score":  round(sem, 4),
                "keyword_score":   round(kw, 4),
                "structure":       structure,
                "language":        lang,
            })

        return results

    def get_file_contents(
        self,
        file_paths: list[str],
        max_chars_per_file: int = 3000,
        truncate: bool = True,
    ) -> dict[str, str]:
        """
        Read the content of specified files from the repository.
        Returns {relative_path: content} with size limits per file.
        Prioritizes relevant sections by truncating from the end.

        Args:
            file_paths: List of relative file path strings.
            max_chars_per_file: Maximum characters to read per file. Must be > 0.

        Returns:
            Dict mapping relative paths to file content strings.

        Raises:
            TypeError: If file_paths is not a list or max_chars_per_file is not an int.
            ValueError: If max_chars_per_file is <= 0.
        """
        if not isinstance(file_paths, list):
            raise TypeError(f"file_paths must be a list, got {type(file_paths).__name__}")
        if not isinstance(max_chars_per_file, int):
            raise TypeError(f"max_chars_per_file must be an int, got {type(max_chars_per_file).__name__}")
        if max_chars_per_file <= 0:
            raise ValueError(f"max_chars_per_file must be > 0, got {max_chars_per_file}")

        contents = {}
        repo_real = os.path.realpath(self.repo_path)

        for rel_path in file_paths:
            # Skip non-string entries
            if not isinstance(rel_path, str) or not rel_path.strip():
                continue

            full_path = os.path.join(self.repo_path, rel_path)
            full_path = os.path.normpath(full_path)

            # Path traversal guard: ensure resolved path is inside repo
            full_real = os.path.realpath(full_path)
            if not full_real.startswith(repo_real):
                print(f"   [BLOCKED] Path traversal attempt: {rel_path}")
                continue

            if not os.path.exists(full_path):
                print(f"   [WARN] File not found: {rel_path}")
                continue

            if not os.path.isfile(full_path):
                print(f"   [WARN] Not a regular file: {rel_path}")
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                if truncate and len(source) > max_chars_per_file:
                    # Smart truncation: always preserve the file header (imports/class signatures)
                    # then fill remaining budget with the tail (where most class bodies live)
                    header_budget = max_chars_per_file // 3
                    tail_budget   = max_chars_per_file - header_budget
                    header = source[:header_budget]
                    tail   = source[-tail_budget:]
                    source = header + "\n# ... (middle truncated) ...\n" + tail
                contents[rel_path] = source
            except OSError as e:
                print(f"   [WARN] Failed to read {rel_path}: {e}")

        return contents

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def print_candidates(self, candidates: list[dict]):
        """
        Display the top candidate files in a readable format.

        Args:
            candidates: List of candidate dicts with path, combined_score, etc.
        """
        if not isinstance(candidates, list) or not candidates:
            print("\n[INFO] No candidates to display.")
            return

        print("\n[TOP] Candidate files:")
        for i, c in enumerate(candidates, 1):
            if not isinstance(c, dict):
                continue
            path = c.get("path", "(unknown)")
            combined = c.get("combined_score", 0)
            sem = c.get("semantic_score", 0)
            kw = c.get("keyword_score", 0)
            print(f"  [{i}] {path}")
            print(f"       score={combined}  (sem={sem}, kw={kw})")
            structure = c.get("structure", {})
            if isinstance(structure, dict):
                fns = structure.get("functions", [])
                if isinstance(fns, list) and fns:
                    print(f"       functions: {', '.join(str(f) for f in fns[:6])}")
        print()
