import os
import sys
from dotenv import load_dotenv

# Ensure modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prevent git checkout issues related to pycache pollution
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from langgraph_flow.graph import build_graph
from langgraph_flow.state import AgentState, PipelineStatus

import requests

def validate_api_keys_at_startup():
    """
    Before touching the repository, cloning branches, or calling any planner,
    verify that all required API keys and credentials are present and valid.
    """
    print("[STARTUP] Synchronizing environments from .env.master...")
    try:
        import subprocess
        subprocess.run([sys.executable, "sync_envs.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[FATAL] Environment synchronization failed: {e}")
        sys.exit(1)
        
    load_dotenv()
    
    # Check GitHub Token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("[FATAL] API key validation failed for GitHub: Missing token")
        print("\nSUMMARY OF FAILURES:")
        print(" - GITHUB_TOKEN is missing. Expected in the root `.env` file.")
        sys.exit(1)

    # Check Groq Token
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("[FATAL] API key validation failed for Groq: Missing key")
        print("\nSUMMARY OF FAILURES:")
        print(" - GROQ_API_KEY is missing. Expected in the root `.env` file.")
        sys.exit(1)

    # Perform lightweight LLM validation call
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1
            },
            timeout=10
        )
        if response.status_code == 401 or response.status_code == 403:
            print(f"[FATAL] API key validation failed for Groq: {response.text}")
            print("\nSUMMARY OF FAILURES:")
            print(" - GROQ_API_KEY is invalid (auth error). Please verify the token in your root `.env` file.")
            sys.exit(1)
        
        # We also check for 200 explicitly 
        response.raise_for_status()
        print("[STARTUP] API Keys Validated Successfully.")
    except SystemExit:
        raise
    except requests.exceptions.RequestException as e:
        print(f"[FATAL] API key validation failed for Groq: {e}")
        print("\nSUMMARY OF FAILURES:")
        print(" - Unable to reach Groq API or generic failure. Please check your network or token configuration.")
        sys.exit(1)

def main():
    validate_api_keys_at_startup()
    print("=" * 60)
    print("  MULTI-AGENT SYSTEM: ISSUE -> QA -> PR")
    print("=" * 60)
    
    graph = build_graph()
    
    # We invoke it with a minimal initial state. Over execution, fetch_issues -> select_issue will populate.
    initial_state = AgentState(status=PipelineStatus.INIT).model_dump()
    
    try:
        # Pass a standard thread ID to use the MemorySaver checkpointer
        config = {"configurable": {"thread_id": "1"}}
        final_state = graph.invoke(initial_state, config=config)
        print("\n\n" + "=" * 60)
        print("  WORKFLOW COMPLETED")
        print("=" * 60)
        print(f"Final Status: {final_state.get('status')}")
        print(f"Retries: {final_state.get('retry_count')}")
        
    except Exception as e:
        print(f"\nPipeline Exception: {e}")

if __name__ == "__main__":
    main()
