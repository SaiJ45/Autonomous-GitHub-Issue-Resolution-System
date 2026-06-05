import os
import time
import requests
from typing import List, Dict, Any

GITHUB_API_BASE = "https://api.github.com"
MAX_RETRIES = 1  # Retry once on failure


def get_headers() -> Dict[str, str]:
    """Build GitHub API headers with authentication if available."""
    token = os.environ.get("GITHUB_TOKEN")

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    if token and token not in ("your_token", "your_github_token_here", "your_actual_api_key_here", ""):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_open_prs(repo: str) -> List[Dict[str, Any]]:
    """Fetch all open pull requests for a repository."""
    all_prs = []
    page = 1
    
    while True:
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls?state=open&per_page=100&page={page}"
        response = requests.get(url, headers=get_headers(), timeout=30)
    
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch open PRs on page {page}. Status: {response.status_code}, Response: {response.text[:300]}")
    
        prs_page = response.json()
        if not prs_page:
            break
            
        all_prs.extend(prs_page)
        page += 1

    filtered_prs = []
    
    for pr in all_prs:
        # 1. Branch name filter (PRIMARY)
        branch_name = pr.get("head", {}).get("ref", "")
        if branch_name.startswith(("qa/", "qa-review/", "qa/review-", "qa/rejection-")):
            continue
            
        # 2. Title filter
        title = pr.get("title", "")
        if title.startswith("QA Review:") or title.startswith("QA Rejection:") or title.startswith("QA Agent:"):
            continue
            
        # 3. Author filter
        user_login = pr.get("user", {}).get("login", "").lower()
        if "bot" in user_login or "qa" in user_login or "agent" in user_login:
            continue
            
        # 4. Label filter
        labels = [lbl.get("name", "").lower() for lbl in pr.get("labels", [])]
        if "qa-generated" in labels:
            continue
            
        filtered_prs.append(pr)
        
    return filtered_prs


def fetch_pr_details(repo: str, pr_number: int) -> Dict[str, Any]:
    """Fetch detailed information about a specific PR."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=get_headers(), timeout=30)

    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PR details. Status: {response.status_code}, Response: {response.text[:300]}")

    data = response.json()
    return {
        "base_branch": data["base"]["ref"],
        "head_branch": data["head"]["ref"],
        "head_repo_clone_url": data["head"]["repo"]["clone_url"],
        "title": data["title"],
        "body": data["body"],
        "diff_url": data["diff_url"]
    }


def fetch_pr_diff(repo: str, pr_number: int) -> str:
    """Fetch the raw diff for a PR."""
    headers = get_headers()
    headers["Accept"] = "application/vnd.github.v3.diff"
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PR diff. Status: {response.status_code}")

    return response.text


def get_modified_files_from_diff(diff_text: str) -> List[str]:
    """Extract modified file paths from a unified diff."""
    files = []
    for line in diff_text.split('\n'):
        if line.startswith('+++ b/'):
            files.append(line[6:])
    return files


def _api_call_with_retry(method: str, url: str, headers: dict, json_payload: dict = None, timeout: int = 30) -> requests.Response:
    """
    Execute an API call with one retry on failure.
    Returns the Response object. Raises on network errors after retry.
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if method == "POST":
                response = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)

            if response.status_code < 300:
                return response

            # Non-2xx — retry if we haven't exhausted attempts
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if attempt < MAX_RETRIES:
                print(f"[API_RETRY] Attempt {attempt + 1} failed ({response.status_code}). Retrying in 2s...")
                time.sleep(2)
            else:
                return response  # Return the failed response after final attempt

        except requests.RequestException as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"[API_RETRY] Network error on attempt {attempt + 1}: {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                raise

    # Should not reach here, but safety net
    raise requests.RequestException(f"API call failed after {MAX_RETRIES + 1} attempts: {last_error}")


def create_pr(repo: str, title: str, body: str, head: str, base: str) -> Dict[str, Any]:
    """
    Creates a pull request using the GitHub API.
    Returns: {"status_code": int, "pr_url": str | None, "error": str | None}

    Anti-hallucination: Only returns pr_url if the API actually returned 201.
    Retries once on failure.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls"
    payload = {
        "title": title,
        "body": body,
        "head": head,
        "base": base
    }

    try:
        response = _api_call_with_retry("POST", url, headers=get_headers(), json_payload=payload)
    except requests.RequestException as e:
        return {"status_code": -1, "pr_url": None, "error": f"Network error after retry: {str(e)}"}

    if response.status_code == 201:
        data = response.json()
        pr_url = data.get("html_url")
        if not pr_url:
            return {"status_code": 201, "pr_url": None, "error": "API returned 201 but no html_url in response."}
        return {"status_code": 201, "pr_url": pr_url, "error": None}
    else:
        return {"status_code": response.status_code, "pr_url": None, "error": response.text[:500]}


def comment_on_pr(repo: str, pr_number: int, body: str) -> Dict[str, Any]:
    """
    Posts a comment on a pull request using the GitHub Issues API.
    Endpoint: POST /repos/{owner}/{repo}/issues/{pr_number}/comments
    Returns: {"status_code": int, "comment_url": str | None, "error": str | None}
    Retries once on failure.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{pr_number}/comments"
    payload = {"body": body}

    try:
        response = _api_call_with_retry("POST", url, headers=get_headers(), json_payload=payload)
    except requests.RequestException as e:
        return {"status_code": -1, "comment_url": None, "error": f"Network error after retry: {str(e)}"}

    if response.status_code == 201:
        data = response.json()
        return {"status_code": 201, "comment_url": data.get("html_url"), "error": None}
    else:
        return {"status_code": response.status_code, "comment_url": None, "error": response.text[:500]}
