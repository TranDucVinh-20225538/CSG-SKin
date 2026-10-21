#!/usr/bin/env python3
"""CPU test: --isic_split_seed 42 reproduces the frozen Table 1 lesion split (path hashes)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r6_split_stability as S  # noqa: E402


def test_seed42_matches_ref():
    ok, msg = S.verify_seed42(strict=False)
    assert ok, msg


if __name__ == "__main__":
    test_seed42_matches_ref()
    print("PASS: seed 42 split hashes match REF_HASHES.json")
