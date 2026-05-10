#!/usr/bin/env python3
"""Run the CPU-only residue introspection POC.

From a fresh clone, without installing BoltzGen's CUDA-heavy dependencies:

    cd boltzgen
    python example/introspective_acceptance_poc.py

With real extracted distributions:

    python example/introspective_acceptance_poc.py --json p_q_distributions.json

JSON can be either:

    {"p": {"A": 0.1, ...}, "q": {"A": 0.2, ...}}

or:

    {
      "residues": [
        {"id": "B:42", "p": {"A": ...}, "q": {"A": ...}},
        {"id": "B:43", "p": {"A": ...}, "q": {"A": ...}}
      ]
    }

This is only a wiring/math check. Real integration still needs `q` from a
BoltzGen proposal head and `p` from an anchor/verifier head or inverse-folding
model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boltzgen.experimental.introspective_acceptance import (  # noqa: E402
    AMINO_ACIDS,
    acceptance_rate,
    exact_final_distribution,
    l1_distance,
    run_simulation,
    toy_pocket_distributions,
)


def _load_cases(path: Path) -> list[tuple[str, dict[str, float], dict[str, float]]]:
    with path.open() as handle:
        payload = json.load(handle)

    if "residues" in payload:
        return [
            (str(entry.get("id", idx)), entry["p"], entry["q"])
            for idx, entry in enumerate(payload["residues"])
        ]

    return [(str(payload.get("id", "single_residue")), payload["p"], payload["q"])]


def _print_case(name: str, p: dict[str, float], q: dict[str, float]) -> None:
    exact = exact_final_distribution(p, q)
    simulation = run_simulation(p, q, n_samples=100_000, seed=7)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Exact expected acceptance rate: {acceptance_rate(p, q):.3f}")
    print(f"Monte Carlo acceptance rate:    {simulation.acceptance_rate:.3f}")
    print(f"L1(exact final, anchor p):      {l1_distance(exact, p):.6f}")
    print(f"L1(MC final, anchor p):         {simulation.l1_to_anchor:.6f}")

    print("\nTop anchor/proposal/final residues:")
    top = sorted(AMINO_ACIDS, key=lambda aa: p[aa], reverse=True)[:8]
    for aa in top:
        print(
            f"  {aa:>2}  p={p[aa]:.3f}  q={q[aa]:.3f}  "
            f"mc_final={simulation.empirical_distribution[aa]:.3f}"
        )

    print("\nCommon rejection corrections:")
    for (rejected, corrected), count in simulation.common_corrections:
        print(f"  {rejected:>2} -> {corrected:>2}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional JSON file containing real p/q distributions to evaluate.",
    )
    args = parser.parse_args()

    p, q = toy_pocket_distributions()
    print("BoltzGen introspective acceptance POC")
    print("=" * 39)

    if args.json is None:
        print("Toy setup: q over-proposes Lys; anchor p prefers Arg/Gln.")
        cases = [("toy residue 42 pocket", p, q)]
    else:
        print(f"Loaded p/q distributions from {args.json}")
        cases = _load_cases(args.json)

    for name, anchor_p, proposal_q in cases:
        _print_case(name, anchor_p, proposal_q)

    print("\nNext plug-in points:")
    print("  q: BoltzGen residue proposal logits over amino acids")
    print("  p: inverse-folding/anchor-mode residue verifier logits")


if __name__ == "__main__":
    main()
