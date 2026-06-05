import re
import requests

try:
    from ..config import REPO_NAME, GITHUB_TOKEN
except ImportError:
    from config import REPO_NAME, GITHUB_TOKEN


# Maximum pages to fetch (safety cap to prevent infinite loops)
_MAX_PAGES = 10
_PER_PAGE = 100


def _build_headers() -> dict:
    """Build GitHub API request headers."""
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def _get_next_page_url(response: requests.Response) -> str | None:
    """
    Extract the 'next' page URL from GitHub's Link header.

    GitHub paginates via Link headers like:
      <https://api.github.com/...?page=2>; rel="next", <...>; rel="last"

    Returns:
        The next page URL string, or None if there is no next page.
    """
    link_header = response.headers.get("Link", "")
    if not link_header:
        return None

    # Find the URL tagged with rel="next"
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _fetch_all_pages(start_url: str, headers: dict) -> list[dict]:
    """
    Fetch ALL pages of results from the GitHub API.

    Follows the Link header pagination until no more pages exist,
    with a safety cap of _MAX_PAGES to prevent runaway loops.

    Args:
        start_url: The initial API URL (page 1).
        headers: Request headers dict.

    Returns:
        Combined list of all items across all pages.

    Raises:
        RuntimeError: If the API returns an unexpected format.
        requests.RequestException: On network/HTTP errors.
    """
    all_items: list[dict] = []
    url: str | None = start_url

    for page_num in range(1, _MAX_PAGES + 1):
        if url is None:
            break

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        items = response.json()
        if not isinstance(items, list):
            message = "Unexpected GitHub issues response format."
            if isinstance(items, dict) and items.get("message"):
                message = f"{message} GitHub said: {items['message']}"
            raise RuntimeError(message)

        if not items:
            # Empty page — no more results
            break

        all_items.extend(items)
        print(f"   [FETCH] Page {page_num}: {len(items)} items (total so far: {len(all_items)})")

        # Check for next page
        url = _get_next_page_url(response)

        # If this page returned fewer than per_page items, it's the last page
        if len(items) < _PER_PAGE:
            break

    return all_items


def get_issues() -> list[dict]:
    """
    Fetch ALL open issues from the GitHub repository.

    Handles pagination to ensure no issues are missed — the GitHub Issues
    API returns both issues and pull requests, so results are filtered
    to exclude PRs. Multiple pages are fetched if necessary.

    Returns:
        List of issue dicts (pull requests excluded).

    Raises:
        RuntimeError: If the API call fails after retries.
    """
    url = (
        f"https://api.github.com/repos/{REPO_NAME}/issues"
        f"?state=open&per_page={_PER_PAGE}&sort=created&direction=asc"
    )
    headers = _build_headers()

    try:
        all_items = _fetch_all_pages(url, headers)
    except requests.RequestException as exc:
        response = getattr(exc, "response", None)

        # Retry without auth if token was rejected
        if (
            response is not None
            and response.status_code == 401
            and GITHUB_TOKEN
        ):
            try:
                print(
                    "[WARN] GitHub token was rejected. "
                    "Retrying issue fetch without authentication..."
                )
                no_auth_headers = {"Accept": "application/vnd.github+json"}
                all_items = _fetch_all_pages(url, no_auth_headers)
            except requests.RequestException:
                pass
            else:
                print(
                    "[WARN] Continuing without GitHub auth. "
                    "Push/PR actions may still fail until GITHUB_TOKEN is fixed."
                )
                issues = [
                    item for item in all_items
                    if "pull_request" not in item
                ]
                print(f"[OK] Retrieved {len(issues)} issue(s) (filtered from {len(all_items)} items)")
                return issues

        detail = ""
        if response is not None:
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("message"):
                    detail = f" GitHub said: {payload['message']}"
            except Exception:
                pass
        raise RuntimeError(
            f"Failed to fetch GitHub issues from {REPO_NAME}.{detail}"
        ) from exc

    # Filter out pull requests (GitHub Issues API includes them)
    issues = [item for item in all_items if "pull_request" not in item]

    print(f"[OK] Retrieved {len(issues)} issue(s) (filtered from {len(all_items)} items)")

    return issues
