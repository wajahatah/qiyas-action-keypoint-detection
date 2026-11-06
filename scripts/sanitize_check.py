"""Publication gate: fail if anything private could leave this repo.

Runs in pre-commit and CI. It scans every file git would publish (tracked + untracked-not-ignored)
and fails on:

* secrets / infrastructure: RTSP URLs, IPv4 addresses, AWS access-key ids, S3 / presigned URLs,
  private-key material, connection strings;
* local paths that identify machines or people (``C:/Users/...``, ``/home/<user>``, ``D:/...``);
* data and weights: images/videos outside the figure allow-list, model weights, office documents,
  CSVs outside ``results/``, notebooks with outputs;
* any name on the deny-list (``scripts/deny_terms.txt``, git-excluded, one term per line) — used
  for colleague and machine names that must never appear.

Usage: ``python scripts/sanitize_check.py``  (exit code 1 on any finding).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONTENT_RULES: dict[str, re.Pattern[str]] = {
    "rtsp-url": re.compile(r"rtsp://", re.I),
    "ipv4": re.compile(
        r"(?<![\w.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\w.])"
    ),
    "aws-key-id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "s3-url": re.compile(r"amazonaws\.com", re.I),
    "presigned": re.compile(r"X-Amz-(?:Credential|Signature)", re.I),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "conn-string": re.compile(r"\b(?:mongodb|redis|amqp|postgres(?:ql)?)://", re.I),
    "windows-user-path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]", re.I),
    "drive-path": re.compile(r"\b[D-Zd-z]:[\\/]+\w"),
    "home-path": re.compile(r"/home/\w+"),
}

# Files allowed to contain rule *definitions* (this file) are exempt from content rules.
CONTENT_EXEMPT = {"scripts/sanitize_check.py"}

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
WEIGHT_EXT = {".pt", ".pth", ".onnx", ".engine", ".h5", ".joblib", ".pkl", ".ckpt", ".tflite"}
OFFICE_EXT = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".pdf"}
FIGURE_DIR = "docs/figures/"
MAX_BYTES = 2_000_000


def publishable_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({p for p in out.splitlines() if p and (ROOT / p).is_file()})


def deny_terms() -> list[str]:
    f = ROOT / "scripts" / "deny_terms.txt"
    if not f.exists():
        return []
    return [t.strip() for t in f.read_text(encoding="utf8").splitlines() if t.strip()]


def check(paths: list[str]) -> list[str]:
    findings: list[str] = []
    terms = deny_terms()
    for rel in paths:
        p = ROOT / rel
        ext = p.suffix.lower()
        if ext in WEIGHT_EXT:
            findings.append(f"{rel}: model weights must not be published")
        if ext in VIDEO_EXT:
            findings.append(f"{rel}: videos must not be published")
        if ext in OFFICE_EXT:
            findings.append(f"{rel}: office documents must not be published")
        if ext in IMAGE_EXT and not rel.startswith(FIGURE_DIR):
            findings.append(f"{rel}: images allowed only under {FIGURE_DIR} (generated figures)")
        if ext == ".csv" and not rel.startswith("results/"):
            findings.append(f"{rel}: CSV data allowed only under results/ (curated metrics)")
        if p.stat().st_size > MAX_BYTES:
            findings.append(f"{rel}: larger than {MAX_BYTES} bytes")
        if ext in IMAGE_EXT:
            continue
        try:
            text = p.read_text(encoding="utf8")
        except (UnicodeDecodeError, OSError):
            findings.append(f"{rel}: unreadable/binary file")
            continue
        if ext == ".ipynb":
            nb = json.loads(text)
            if any(c.get("outputs") for c in nb.get("cells", [])):
                findings.append(f"{rel}: notebook has outputs (strip them)")
        if rel not in CONTENT_EXEMPT:
            for name, rx in CONTENT_RULES.items():
                for m in rx.finditer(text):
                    line = text.count("\n", 0, m.start()) + 1
                    findings.append(f"{rel}:{line}: {name}")
        low = text.lower()
        for t in terms:
            if t.lower() in low or t.lower() in rel.lower():
                findings.append(f"{rel}: deny-listed term (see scripts/deny_terms.txt)")
    return findings


def main() -> int:
    findings = check(publishable_files())
    if findings:
        print("Sanitisation gate FAILED:")
        for f in findings:
            print("  -", f)
        return 1
    print("Sanitisation gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
