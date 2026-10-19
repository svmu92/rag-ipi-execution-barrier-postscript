"""Shared helpers for the scripts/ CLI entry points: config loading, result
logging and the provenance fields the plan's working agreements require
(section 4 step 7: "Log the git commit hash, requirements.lock.txt hash and
config with every result line.").
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `import src...` works when run as a script


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git-repo"


def requirements_lock_hash() -> str:
    lock_path = REPO_ROOT / "requirements.lock.txt"
    if not lock_path.exists():
        return "no-lock-file"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()[:12]


def provenance() -> dict:
    return {"git_commit": git_commit_hash(), "requirements_lock_hash": requirements_lock_hash()}


class JsonlWriter:
    """Appends one JSON object per line, each tagged with config + provenance
    (plan section 4 step 7). Never edit a results file by hand -- regenerate
    tables/figures from these with scripts/07_analyze.py."""

    def __init__(self, path: str | Path, config: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.prov = provenance()

    def write(self, record: dict) -> None:
        row = {**record, "_config": self.config, "_provenance": self.prov}
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")  # ensure_ascii: see README on invisible chars


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
