"""
sync_envs.py — keeps all three .env files in sync from .env.master
Run this once before starting the agent pipeline.
"""
import os

MASTER = ".env.master"

TARGETS = {
    "agents/ai_issue_agent/.env": [
        "GROQ_API_KEY", "GITHUB_TOKEN", "REPO_NAME", "REPO_OWNER"
    ],
    ".env": [
        "GROQ_API_KEY", "GITHUB_TOKEN", "GITHUB_REPO"
    ],
}

def load_master(path):
    kv = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()
    return kv

def write_env(path, keys, master):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    missing = [k for k in keys if k not in master]
    if missing:
        raise ValueError(f"Missing in master: {missing}")
    with open(path, "w") as f:
        for k in keys:
            f.write(f"{k}={master[k]}\n")
    print(f"[OK] Written: {path} ({len(keys)} keys)")

def validate_env(path, keys):
    from dotenv import dotenv_values
    env_content = dotenv_values(path)
    for k in keys:
        if not env_content.get(k):
            raise ValueError(f"Validation failed: Key {k} missing or empty in {path}")
    print(f"[OK] Validated: {path}")

def main():
    if not os.path.exists(MASTER):
        raise FileNotFoundError(f"{MASTER} not found.")
        
    master = load_master(MASTER)
    print("--- Syncing Environments ---")
    for target_path, allowlist in TARGETS.items():
        write_env(target_path, allowlist, master)
        validate_env(target_path, allowlist)
    print("--- Done ---")

if __name__ == "__main__":
    main()
