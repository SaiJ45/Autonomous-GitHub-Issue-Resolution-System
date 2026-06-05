import os
from dotenv import load_dotenv

def load_environment():
    load_dotenv(override=True)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    github_token = os.getenv("GITHUB_TOKEN")
    github_repo = os.getenv("GITHUB_REPO")
    
    if not groq_api_key or groq_api_key.strip() == "" or groq_api_key == "your_token":
        raise ValueError("GROQ_API_KEY is missing or not configured correctly in the .env file.")
        
    if not github_repo or github_repo.strip() == "" or github_repo == "owner/repo":
        raise ValueError("GITHUB_REPO is missing or not configured correctly in the .env file. Format must be owner/repo")
        
    return {
        "GROQ_API_KEY": groq_api_key,
        "GITHUB_TOKEN": github_token,
        "GITHUB_REPO": github_repo
    }

# Automatically execute on import
config_vars = load_environment()
