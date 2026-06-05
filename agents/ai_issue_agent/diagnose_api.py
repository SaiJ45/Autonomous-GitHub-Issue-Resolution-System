"""
diagnose_api.py — Run this to debug your Groq API key issue.
Usage:  .venv\Scripts\python.exe diagnose_api.py
"""
import os
import sys
from pathlib import Path

print("=" * 55)
print("GROQ API KEY DIAGNOSTIC")
print("=" * 55)

# --- Step 1: Check what's in the system environment ---
sys_key = os.environ.get("GROQ_API_KEY")
if sys_key:
    print(f"\n[1] ⚠️  Windows SYSTEM env var GROQ_API_KEY is SET: {sys_key[:12]}...{sys_key[-4:]}")
    print("    This was silently overriding your .env file before the fix!")
else:
    print("\n[1] ✅ No stale GROQ_API_KEY in Windows system environment")

# --- Step 2: Load from .env with override ---
try:
    from dotenv import load_dotenv
except ImportError:
    print("[2] ERROR: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

env_path = Path(".env")
if not env_path.exists():
    print(f"[2] ERROR: .env file not found at {env_path.absolute()}")
    sys.exit(1)

# Print raw .env content (mask middle of key)
print(f"\n[2] .env file found at: {env_path.absolute()}")
print("    Contents:")
for line in env_path.read_text().splitlines():
    if "KEY" in line or "TOKEN" in line:
        k, _, v = line.partition("=")
        masked = v[:8] + "..." + v[-4:] if len(v) > 12 else v
        print(f"      {k}={masked}")
    else:
        print(f"      {line}")

load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("GROQ_API_KEY")

print(f"\n[3] Key loaded after override: {key[:12]}...{key[-4:] if key else ''}")
print(f"    Length: {len(key) if key else 0}  (expected ~56 chars)")

if not key:
    print("[3] ERROR: Key is empty/None after loading")
    sys.exit(1)

if key.startswith('"') or key.startswith("'"):
    print("[3] ERROR: Key has surrounding quotes in .env — remove them!")
    sys.exit(1)

if key != key.strip():
    print("[3] ERROR: Key has extra whitespace in .env — clean it!")
    sys.exit(1)

if not key.startswith("gsk_"):
    print(f"[3] ERROR: Key doesn't start with 'gsk_' — may be malformed")
    sys.exit(1)

print("    Format looks correct ✅")

# --- Step 3: Make a real API call ---
print("\n[4] Testing Groq API call...")
try:
    from groq import Groq
    client = Groq(api_key=key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Reply with just the word: OK"}],
        max_tokens=5,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"    ✅ SUCCESS! Model replied: '{answer}'")
    print("\n✅ Your API key is working. Run the agent normally.")
except Exception as e:
    print(f"    ❌ API call failed: {e}")
    print()
    print("Troubleshoot:")
    print("  • If '401 Invalid API Key' → Go to console.groq.com, delete and recreate the key")
    print("  • If '429 Rate Limit'      → Wait 1-2 minutes and retry")
    print("  • If 'Connection Error'    → Check your internet / proxy / VPN")
    print("  • If model not found       → Try model='llama3-8b-8192' (free tier always available)")
