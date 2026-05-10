#!/usr/bin/env python3
"""Probe whether a BoltzGen checkpoint can expose proposal q logits.

For the introspective-acceptance idea, the design/proposal model needs a
residue-level distribution q, ideally `res_type` logits from the diffusion
proposal path. This script checks the actual checkpoint for:

* hyperparameter `predict_res_type`
* state_dict parameters for a residue classifier head

It does not run inference and can be run on CPU after the checkpoint is
downloaded:

    boltzgen download design-diverse inverse-fold
    python example/probe_design_q_logits.py --checkpoint /path/to/boltzgen1_diverse.ckpt
    python example/probe_design_q_logits.py --checkpoint /path/to/boltzgen1_ifold.ckpt

If `predict_res_type=false` and no residue classifier weights are present,
the design checkpoint cannot provide q logits without adding/training a head.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _find(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def probe(checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    state_dict = ckpt.get("state_dict", ckpt)

    predict_res_type = _find(hparams, "predict_res_type")
    inverse_fold = _find(hparams, "inverse_fold")
    design_head_keys = [
        key
        for key in state_dict
        if "res_type_predictor" in key or key.endswith(".res_type")
    ]
    inverse_fold_head_keys = [
        key
        for key in state_dict
        if key.startswith("structure_module.predictor.")
    ]
    head_keys = design_head_keys + inverse_fold_head_keys

    print(f"checkpoint: {checkpoint}")
    print(f"hyper_parameters.predict_res_type: {predict_res_type}")
    print(f"hyper_parameters.inverse_fold:      {inverse_fold}")
    print(f"design q head weights found:        {bool(design_head_keys)}")
    print(f"inverse-fold p head weights found:  {bool(inverse_fold_head_keys)}")

    if head_keys:
        print("\nmatching state_dict keys:")
        for key in head_keys[:20]:
            shape = tuple(state_dict[key].shape) if hasattr(state_dict[key], "shape") else "?"
            print(f"  {key}: {shape}")
        if len(head_keys) > 20:
            print(f"  ... {len(head_keys) - 20} more")

    print("\ninterpretation:")
    if inverse_fold is True and inverse_fold_head_keys:
        print("  This looks like an inverse-folding anchor p checkpoint, not design q.")
    elif predict_res_type is True and design_head_keys:
        print("  This design checkpoint likely can expose proposal q residue logits.")
    else:
        print("  This checkpoint likely does not expose q logits; add/train a q head.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    probe(args.checkpoint)


if __name__ == "__main__":
    main()
