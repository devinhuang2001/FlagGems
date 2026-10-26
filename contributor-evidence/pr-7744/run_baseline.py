"""Swap only the PR's production file, validate, and always restore candidate."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--baseline-source", default="415ea38c63141382b0051eac5ef0fa22cbbbec69")
args = parser.parse_args()
repo = args.repo.resolve()
evidence = Path(__file__).resolve().parent
source = repo / "src/flag_gems/ops/eq.py"
baseline_ref = args.baseline_source
candidate = source.read_bytes()
baseline = subprocess.check_output(
    ["git", "show", f"{baseline_ref}:src/flag_gems/ops/eq.py"], cwd=repo
)
result = {"baseline_eq_source": baseline_ref,
          "candidate_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
try:
    source.write_bytes(baseline)
    commands = {
        "target": ["-m", "pytest", "-q", "tests/test_eq_integer_precision.py", "tests/test_eq.py",
                   "tests/test_equal.py", f"--junitxml={evidence / 'baseline-target.xml'}"],
        "promotion": [str(evidence / "promotion_matrix.py")],
    }
    for name, arguments in commands.items():
        suffix = "json" if name == "promotion" else "log"
        with (evidence / f"baseline-{name}.{suffix}").open("w", encoding="utf-8") as output:
            result[f"{name}_exit"] = subprocess.run(
                [sys.executable, *arguments], cwd=repo, stdout=output, stderr=subprocess.STDOUT
            ).returncode
finally:
    source.write_bytes(candidate)
    result["candidate_restored"] = source.read_bytes() == candidate
    result["candidate_source_sha256"] = hashlib.sha256(candidate).hexdigest()
    (evidence / "baseline-control.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
