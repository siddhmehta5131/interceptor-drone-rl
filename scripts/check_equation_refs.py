"""Check equation IDs in docs/mathematics.md against code and configs (P2.2)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID_RE = re.compile(r"\bEQ-[A-Z][A-Z0-9]*-\d+\b")


def collect_ids(text: str) -> set:
    return set(ID_RE.findall(text))


def scan_files(root: Path) -> set:
    refs = set()
    for pattern in ("interceptor/**/*.py", "configs/**/*.yaml", "scripts/**/*.py"):
        for path in root.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            refs |= collect_ids(path.read_text(encoding="utf-8", errors="ignore"))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--math", default="docs/mathematics.md")
    args = parser.parse_args()
    root = Path(args.root)
    doc_ids = collect_ids((root / args.math).read_text(encoding="utf-8"))
    refs = scan_files(root)
    unknown = sorted(refs - doc_ids)
    unlinked = sorted(doc_ids - refs)
    if unknown:
        print("Unknown equation references (not defined in mathematics.md):")
        for item in unknown:
            print(f"  {item}")
    if unlinked:
        print("Equation IDs defined but never cited in code/config:")
        for item in unlinked:
            print(f"  {item}")
    if unknown or unlinked:
        print(f"FAIL: {len(unknown)} unknown, {len(unlinked)} unlinked")
        return 1
    print(f"PASS: {len(doc_ids)} equation IDs, all cited")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
