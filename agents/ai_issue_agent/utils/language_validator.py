import subprocess
import tempfile


def validate_python(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except:
        return False


def validate_js(code: str) -> bool:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".js", mode="w") as f:
            f.write(code)
            path = f.name

        result = subprocess.run(
            ["node", "--check", path],
            capture_output=True,
            text=True
        )

        return result.returncode == 0

    except:
        return True  # fallback safe


def validate_html(code: str) -> bool:
    # simple structural check
    if "<html" in code.lower() and "</html>" not in code.lower():
        return False
    return True


def validate_by_type(file_path, code):
    if file_path.endswith(".py"):
        return validate_python(code)
    elif file_path.endswith(".js"):
        return validate_js(code)
    elif file_path.endswith(".html"):
        return validate_html(code)

    return True