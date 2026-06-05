import logging
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log

logger = logging.getLogger(__name__)


class TokenLimitError(RuntimeError):
    """Raised when an LLM request exceeds the token/context limit (HTTP 413)."""
    pass


def is_token_limit_error(exception: Exception) -> bool:
    """Return True if this error is a per-request token/context-length limit (413)."""
    error_str = str(exception)
    checks = (
        "413" in error_str,
        "request_too_large" in error_str.lower(),
        "too large" in error_str.lower(),
        "maximum context length" in error_str.lower(),
        "tokens_exceeded" in error_str.lower(),
        "context_length_exceeded" in error_str.lower(),
        "TOKEN_LIMIT_EXCEEDED" in error_str,
    )
    return any(checks)


def is_retriable_llm_error(exception: Exception) -> bool:
    """Determine whether the exception is worth retrying.

    Non-retriable: daily quota exhaustion, auth errors, 413 token-limit errors.
    Retriable: transient 429 rate limits and other temporary failures.
    """
    # Never retry a 413 — same payload will fail again
    if is_token_limit_error(exception):
        return False

    error_str = str(exception)
    if "RATE_LIMIT_DAILY_EXHAUSTED" in error_str:
        return False
    if "TOKEN_LIMIT_EXCEEDED" in error_str:
        return False
    if "401" in error_str or "unauthorized" in error_str.lower():
        return False
    if "403" in error_str or "forbidden" in error_str.lower():
        return False
    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    retry=retry_if_exception(is_retriable_llm_error),
)
def safe_invoke(func, *args, **kwargs):
    """
    Safely invoke the LLM with exponential backoff for transient rate limits.

    Raises TokenLimitError immediately on 413 — caller must reduce prompt size.
    Raises RuntimeError on daily quota exhaustion or auth failures (no retry).
    Supports any callable (LangChain llm.invoke or raw client.chat.completions.create).
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_str = str(e)

        # 413 / token limit — non-retriable; caller must shrink the prompt
        if is_token_limit_error(e):
            logger.error(
                f"[safe_invoke] Token/context limit exceeded (413). "
                f"Reduce prompt size. Detail: {error_str[:200]}"
            )
            raise TokenLimitError(
                f"TOKEN_LIMIT_EXCEEDED — prompt too large: {error_str[:300]}"
            ) from e

        if "429" in error_str:
            try:
                if hasattr(e, "response") and e.response:
                    err_data = e.response.json()
                    err_type = err_data.get("error", {}).get("type", "")
                    if err_type == "tokens":
                        logger.error("Daily token quota exhausted. Will not retry.")
                        raise RuntimeError(
                            "RATE_LIMIT_DAILY_EXHAUSTED — quota will not recover within this run"
                        ) from e
                elif "'type': 'tokens'" in error_str or '"type":"tokens"' in error_str.replace(" ", ""):
                    logger.error("Daily token quota exhausted based on error message. Will not retry.")
                    raise RuntimeError(
                        "RATE_LIMIT_DAILY_EXHAUSTED — quota will not recover within this run"
                    ) from e
            except TokenLimitError:
                raise
            except RuntimeError:
                raise
            except Exception:
                pass

        if "401" in error_str or "unauthorized" in error_str.lower():
            logger.error("Auth error detected (401). Configuration failure.")
            raise RuntimeError(
                f"AUTH_ERROR_401 — Pipeline configured with invalid/missing API key: {error_str[:200]}"
            ) from e

        if "403" in error_str or "forbidden" in error_str.lower():
            logger.error("Auth error detected (403). Configuration failure.")
            raise RuntimeError(
                f"AUTH_ERROR_403 — Pipeline configured with invalid/missing API key: {error_str[:200]}"
            ) from e

        logger.error(f"LLM invoke failed: {str(e)[:500]}")
        raise e
