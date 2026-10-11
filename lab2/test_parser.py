import subprocess
import json
import os
import sys

TEST_DIR = "tests"

def run_parser(input_file):
    try:
        result = subprocess.run(
            [sys.executable, "funny_parser.py", input_file],
            capture_output=True, text=True, timeout=3.0
        )
        try:
            output_json = json.loads(result.stdout)
        except json.JSONDecodeError:
            output_json = None
        return result.returncode, output_json, result.stderr
    except subprocess.TimeoutExpired:
        return -1, None, "PARSER TIMEOUT"

def test_positive():
    positive_dir = os.path.join(TEST_DIR, "positive")
    passed, total = 0, 0
    for filename in sorted(os.listdir(positive_dir)):
        if not filename.endswith(".funny"): continue
        total += 1
        filepath = os.path.join(positive_dir, filename)
        print(f"Testing: {filename:<30}", end=" ")
        code, out, err = run_parser(filepath)
        if code == 0 and out and out.get("errors") == []:
            print("PASS"); passed += 1
        else:
            print("FAIL")
            print(f"  Errors: {out.get('errors') if out else 'N/A'}")
            if err: print(f"  Stderr: {err.strip()}")
    print(f"Positive Tests: {passed}/{total} passed.\n")

def test_negative():
    negative_dir = os.path.join(TEST_DIR, "negative")
    passed, total = 0, 0
    for filename in sorted(os.listdir(negative_dir)):
        if not filename.endswith(".funny"): continue
        total += 1
        filepath = os.path.join(TEST_DIR, filename)
        print(f"Testing: {filename:<30}", end=" ")
        code, out, err = run_parser(filepath)
        if code == -1:
            print("FAIL (INFINITE LOOP / TIMEOUT)")
        elif code != 0 or (out and len(out.get("errors", [])) > 0):
            print("PASS"); passed += 1
        else:
            print("FAIL (wrong code or no errors)")
    print(f"Negative Tests: {passed}/{total} passed.\n")

if __name__ == "__main__":
    test_positive()
    test_negative()