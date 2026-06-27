# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Rosetta fixture generator for sketchir MinHash (DISTRIBUTIONAL class).

Provenance for sketchir_minhash.json.

MinHash estimates the Jaccard similarity of two sets; the estimate is stochastic
(depends on the hash family), so a cross-library hash-for-hash match is
impossible. The DISTRIBUTIONAL check instead uses the EXACT true Jaccard as
ground truth and asserts sketchir's estimate is within sampling error
(~sqrt(J(1-J)/num_hashes) per the MinHash variance). The generator computes the
true Jaccard from the sets so the Rust test does not recompute it (and so cannot
repeat sketchir's own potential error).

Two set pairs with different true Jaccard (~1/3 and ~2/3) check that the
estimator tracks similarity across the range, not just at one point.

Regenerate: uv run tests/fixtures/rosetta/gen_sketchir.py
"""

import json
import platform
from pathlib import Path


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


# Pair 1: A = 0..999, B = 500..1499 -> inter 500, union 1500, J = 1/3.
a1 = list(range(0, 1000))
b1 = list(range(500, 1500))
# Pair 2: A = 0..999, B = 200..1199 -> inter 800, union 1200, J = 2/3.
a2 = list(range(0, 1000))
b2 = list(range(200, 1200))

num_hashes = 256
seed = 42

fixture = {
    "provenance": {
        "generator": "gen_sketchir.py",
        "oracle": "exact set Jaccard (ground truth, not a library)",
        "python": platform.python_version(),
        "note": "DISTRIBUTIONAL: estimate within MinHash sampling error of true J.",
    },
    "num_hashes": num_hashes,
    "seed": seed,
    "pairs": [
        {"a": a1, "b": b1, "true_jaccard": jaccard(a1, b1)},
        {"a": a2, "b": b2, "true_jaccard": jaccard(a2, b2)},
    ],
}

out = Path(__file__).parent / "sketchir_minhash.json"
out.write_text(json.dumps(fixture) + "\n")
for p in fixture["pairs"]:
    print(f"true jaccard {p['true_jaccard']:.6f}")
print(f"wrote {out}")
