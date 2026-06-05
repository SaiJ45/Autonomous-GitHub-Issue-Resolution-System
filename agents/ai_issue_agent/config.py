import os
from pathlib import Path
from dotenv import load_dotenv

# override=True ensures .env always wins over stale Windows system env vars
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


def _clean_env(name: str):
    value = os.getenv(name)
    if value is None:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()

    return value or None


GITHUB_TOKEN = _clean_env("GITHUB_TOKEN")
GROQ_API_KEY = _clean_env("GROQ_API_KEY")
REPO_OWNER   = _clean_env("REPO_OWNER")
REPO_NAME    = _clean_env("REPO_NAME")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. "
        f"Check that {_env_path} exists and contains: GROQ_API_KEY=gsk_..."
    )

if not GROQ_API_KEY.startswith("gsk_"):
    raise EnvironmentError(
        f"GROQ_API_KEY looks malformed (should start with 'gsk_'). "
        f"Got: {GROQ_API_KEY[:12]}..."
    )

CLONE_PATH = "./repo_clone"
