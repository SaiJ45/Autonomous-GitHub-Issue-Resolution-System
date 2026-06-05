import py_compile
import sys

files = [
    "main.py",
    "agents/patch_generator.py",
    "git_tools/clone_repo.py",
    "git_tools/commit_push.py",
    "utils/quality_checker.py",
    "utils/self_reviewer.py",
    "utils/verifier.py",
    "utils/scope_verifier.py",
]

errors = []
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  OK  {f}")
    except py_compile.PyCompileError as e:
        print(f"  FAIL  {f}: {e}")
        errors.append(f)

if errors:
    print(f"\n{len(errors)} file(s) have syntax errors")
    sys.exit(1)
else:
    print(f"\nAll {len(files)} files compiled successfully")
