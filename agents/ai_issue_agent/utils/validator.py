def validate_fix(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # ❌ Merge conflict markers
        if "<<<<<<<" in content or "=======" in content or ">>>>>>>" in content:
            print("[FAIL] Merge conflict markers found")
            return False

        # ❌ Markdown / LLM artifacts
        if "```" in content or "###" in content:
            print("[FAIL] Markdown artifacts detected")
            return False

        # ❌ LLM explanation leakage
        forbidden_phrases = [
            "Explanation",
            "Fixed Code",
            "Here is the corrected",
            "Advice"
        ]

        for phrase in forbidden_phrases:
            if phrase.lower() in content.lower():
                print(f"[FAIL] LLM explanation detected: {phrase}")
                return False

        # ❌ Random hallucinated junk (basic filter)
        if "OLIAHFLAK" in content:
            print("[FAIL] Random hallucinated content detected")
            return False

        # ❌ Broken HTML structure
        if "<html" in content.lower() and "</html>" not in content.lower():
            print("[FAIL] Invalid HTML structure")
            return False

        # ❌ Empty / too small
        if len(content.strip()) < 20:
            print("[FAIL] File too small or empty")
            return False

        return True

    except Exception as e:
        print(f"[FAIL] Validator error: {e}")
        return False