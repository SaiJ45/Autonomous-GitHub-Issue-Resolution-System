import ast
import re

def _extract_python_metadata(content: str) -> str:
    """Extract class/function names and docstrings to enrich chunks."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return ""

    metadata_lines = []
    
    # Extract structural names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            metadata_lines.append(f"Class: {node.name}")
            doc = ast.get_docstring(node)
            if doc:
                metadata_lines.append(f"Docstring: {doc.strip().split(chr(10))[0]}")
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            metadata_lines.append(f"Function: {node.name}")
            
    # Include first few lines of imports as metadata
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
            
    if imports:
        metadata_lines.append(f"Dependencies: {', '.join(imports)}")

    return "\n".join(metadata_lines)

def _extract_generic_metadata(content: str, ext: str) -> str:
    """Fallback metadata extraction for non-Python files using regex."""
    metadata_lines = []
    
    # Generic function/class regexes
    if ext in [".js", ".ts", ".jsx", ".tsx"]:
        funcs = re.findall(r"(?:function|const|let|var)\s+([a-zA-Z_$][0-9a-zA-Z_$]*)\s*(?:=|:|\()", content)
        classes = re.findall(r"class\s+([a-zA-Z_$][0-9a-zA-Z_$]*)", content)
        if funcs:
            metadata_lines.append(f"Functions: {', '.join(set(funcs[:10]))}")
        if classes:
            metadata_lines.append(f"Classes: {', '.join(set(classes[:10]))}")
            
    elif ext in [".html", ".css"]:
        ids = re.findall(r"id=['\"]([^'\"]+)['\"]", content)
        classes = re.findall(r"class=['\"]([^'\"]+)['\"]", content)
        if ids:
            metadata_lines.append(f"IDs: {', '.join(set(ids[:10]))}")
        if classes:
            # Note: naive CSS matching
            metadata_lines.append(f"CSS Classes: {', '.join(set(classes[:10]))}")
            
    return "\n".join(metadata_lines)

def chunk_file(file_path: str, content: str, chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """
    Splits file content into overlapping chunks of approx `chunk_size` words.
    Prepends relevant metadata to each chunk to strengthen embeddings.
    """
    ext = "." + file_path.split(".")[-1].lower() if "." in file_path else ""
    
    # Extract metadata at the file level
    metadata = ""
    if ext == ".py":
        metadata = _extract_python_metadata(content)
    else:
        metadata = _extract_generic_metadata(content, ext)
        
    metadata_prefix = f"File: {file_path}\n"
    if metadata:
        metadata_prefix += metadata + "\n---\n"
        
    tokens = content.split()
    chunks = []
    
    # If file is empty or very small, return single chunk
    if not tokens:
        return [{"chunk_text": metadata_prefix, "file_path": file_path, "chunk_idx": 0}]
        
    idx = 0
    start = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_words = tokens[start:end]
        chunk_text = metadata_prefix + " ".join(chunk_words)
        
        chunks.append({
            "chunk_text": chunk_text,
            "file_path": file_path,
            "chunk_idx": idx
        })
        
        idx += 1
        start += chunk_size - overlap
        
    return chunks
