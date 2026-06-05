import os
import configure
from langchain_groq import ChatGroq

_LLM_INSTANCE = None

def get_llm():
    """
    Returns a highly resilient, globally shared Groq LLM instance perfectly suited 
    for strict deterministic reasoning in the QA pipeline.
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_token":
        raise ValueError("GROQ_API_KEY environment variable is missing or invalid.")
    
    # Primary model configuration
    primary_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0,
        max_tokens=4096,
        max_retries=1
    )
    
    # Fallback model configuration (mixtral)
    fallback_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=4096,
        max_retries=1
    )
    
    # Wrap primary with fallback capabilities
    _LLM_INSTANCE = primary_llm.with_fallbacks([fallback_llm])
    
    return _LLM_INSTANCE

def ping_llm() -> bool:
    """
    Validates infrastructure startup.
    Pings the Groq API using the configured models to guarantee availability before proceeding.
    """
    try:
        llm = get_llm()
        resp = llm.invoke("ping")
        return resp is not None
    except Exception as e:
        print(f"CRITICAL INFRASTRUCTURE FAILURE: LLM missing, misconfigured, or unreachable.\nDetails: {e}")
        return False
