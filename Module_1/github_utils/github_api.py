import requests
import os
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_user_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else []

def get_java_js_files(username, repo_name):
    url = f"https://api.github.com/repos/{username}/{repo_name}/commits"
    commits = requests.get(url, headers=HEADERS).json()
    file_urls = []
    
    for commit in commits[:5]:  # Limit for performance
        sha = commit.get("sha")
        if not sha:
            continue
        commit_data = requests.get(
            f"https://api.github.com/repos/{username}/{repo_name}/commits/{sha}",
            headers=HEADERS
        ).json()
        files = commit_data.get("files", [])
        for f in files:
            if f["filename"].endswith((".java", ".js")):
                file_urls.append({
                    "filename": f["filename"],
                    "raw_url": f.get("raw_url"),
                    "language": "Java" if f["filename"].endswith(".java") else "JavaScript"
                })
    return file_urls

def fetch_code_from_url(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else ""
