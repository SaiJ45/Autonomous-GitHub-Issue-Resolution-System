import os
import re

ALLOWED_EXTENSIONS = (".py", ".js", ".ts", ".html", ".css")


class CodeRetriever:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def get_all_files(self):
        files = []

        for root, _, filenames in os.walk(self.repo_path):
            for f in filenames:
                if f.endswith(ALLOWED_EXTENSIONS):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.repo_path)
                    files.append(rel_path)

        return files

    def read_file(self, file_path: str):
        try:
            with open(os.path.join(self.repo_path, file_path), "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""

    def score_file(self, content: str, issue_text: str):
        """
        Simple keyword scoring
        """

        score = 0

        issue_words = re.findall(r"\w+", issue_text.lower())

        for word in issue_words:
            if word in content.lower():
                score += 1

        return score

    def retrieve(self, issue_text: str, top_k: int = 3):
        """
        Returns top relevant files
        """

        files = self.get_all_files()

        scored = []

        for file_path in files:
            content = self.read_file(file_path)

            if not content.strip():
                continue

            score = self.score_file(content, issue_text)

            if score > 0:
                scored.append((file_path, content, score))

        # sort by score descending
        scored.sort(key=lambda x: x[2], reverse=True)

        top_files = scored[:top_k]

        print("\n🔍 Retrieved Files:")
        for f in top_files:
            print(f" - {f[0]} (score: {f[2]})")

        return [(f[0], f[1]) for f in top_files]