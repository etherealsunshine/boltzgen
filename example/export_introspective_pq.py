#!/usr/bin/env python3
"""Export BoltzGen proposal/anchor residue distributions for the POC.

This script bridges real BoltzGen `.npz` outputs to
`example/introspective_acceptance_poc.py --json`.

Expected inputs:

* proposal NPZ with `res_type_logits` from a BoltzGen design/proposal run.
* anchor NPZ with `inverse_fold_logits` or `logits` from inverse-folding /
  future anchor-mode verification.

Example:

    python example/export_introspective_pq.py \
      --proposal-npz workdir/design/intermediate_designs/foo.npz \
      --anchor-npz workdir/design/intermediate_designs_inverse_folded/foo.npz \
      --out workdir/design/foo_pq.json

Then:

    python example/introspective_acceptance_poc.py --json workdir/design/foo_pq.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


AA = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)

# BoltzGen token order is:
# ["<pad>", "-", ALA, ARG, ..., VAL, "UNK", ...]
CANONICAL_TOKEN_OFFSET = 2


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _pick_key(payload: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in payload:
            return key
    msg = f"none of {candidates} found; available keys: {sorted(payload)}"
    raise KeyError(msg)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _as_residue_probs(values: np.ndarray, *, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        msg = f"{key} must have shape (residues, tokens/classes); got {arr.shape}"
        raise ValueError(msg)

    if arr.shape[-1] >= CANONICAL_TOKEN_OFFSET + len(AA):
        arr = arr[:, CANONICAL_TOKEN_OFFSET : CANONICAL_TOKEN_OFFSET + len(AA)]
    elif arr.shape[-1] != len(AA):
        msg = (
            f"{key} last dimension must be 20 AA classes or full BoltzGen token "
            f"vocab; got {arr.shape[-1]}"
        )
        raise ValueError(msg)

    row_sums = arr.sum(axis=-1)
    looks_like_probs = np.all(arr >= 0) and np.allclose(row_sums, 1.0, atol=1e-3)
    return arr if looks_like_probs else _softmax(arr)


def _mask(payload: dict[str, Any], n_residues: int) -> np.ndarray:
    if "design_mask" not in payload:
        return np.ones(n_residues, dtype=bool)
    mask = np.asarray(payload["design_mask"]).astype(bool)
    if mask.ndim > 1:
        mask = np.reshape(mask, (-1,))
    if mask.shape[0] != n_residues:
        msg = f"design_mask length {mask.shape[0]} does not match {n_residues}"
        raise ValueError(msg)
    return mask


def _token_indices(payload: dict[str, Any], n_residues: int) -> list[int]:
    if "token_index" not in payload:
        return list(range(n_residues))
    token_index = np.asarray(payload["token_index"]).reshape(-1)
    if token_index.shape[0] != n_residues:
        return list(range(n_residues))
    return [int(x) for x in token_index]


def _dist_row(row: np.ndarray) -> dict[str, float]:
    return {aa: float(row[idx]) for idx, aa in enumerate(AA)}


def export_pq(
    proposal_npz: Path,
    anchor_npz: Path,
    out: Path,
    *,
    proposal_key: str | None = None,
    anchor_key: str | None = None,
) -> None:
    proposal = _load_npz(proposal_npz)
    anchor = _load_npz(anchor_npz)

    proposal_key = proposal_key or _pick_key(
        proposal, ("res_type_logits", "proposal_logits", "q", "res_type")
    )
    anchor_key = anchor_key or _pick_key(
        anchor, ("inverse_fold_logits", "anchor_logits", "p", "logits", "res_type_logits")
    )

    q = _as_residue_probs(proposal[proposal_key], key=proposal_key)
    p = _as_residue_probs(anchor[anchor_key], key=anchor_key)
    n = min(len(q), len(p))
    q = q[:n]
    p = p[:n]

    mask = _mask(proposal, n) & _mask(anchor, n)
    token_indices = _token_indices(proposal, n)

    residues = []
    for idx in np.flatnonzero(mask):
        residues.append(
            {
                "id": f"token_{token_indices[idx]}",
                "p": _dist_row(p[idx]),
                "q": _dist_row(q[idx]),
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        json.dump(
            {
                "source": {
                    "proposal_npz": str(proposal_npz),
                    "proposal_key": proposal_key,
                    "anchor_npz": str(anchor_npz),
                    "anchor_key": anchor_key,
                },
                "residues": residues,
            },
            handle,
            indent=2,
        )

    print(f"wrote {len(residues)} residue p/q distributions to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-npz", type=Path, required=True)
    parser.add_argument("--anchor-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposal-key", help="Override proposal NPZ key.")
    parser.add_argument("--anchor-key", help="Override anchor NPZ key.")
    args = parser.parse_args()

    export_pq(
        args.proposal_npz,
        args.anchor_npz,
        args.out,
        proposal_key=args.proposal_key,
        anchor_key=args.anchor_key,
    )


if __name__ == "__main__":
    main()

