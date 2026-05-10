#!/usr/bin/env python3
"""No-data smoke test for the experimental q-head setup.

This script does not construct datasets or run a forward pass. It only checks
that the q-head training configuration can:

1. instantiate a design model with `predict_res_type=true`;
2. load a released design checkpoint non-strictly;
3. leave only `res_type_predictor` parameters trainable.

Run from the BoltzGen repo root:

    python example/smoke_q_head_no_data.py \
      --checkpoint /path/to/boltzgen1_diverse.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch


def _matches_any(name: str, substrings: tuple[str, ...]) -> bool:
    return any(part in name for part in substrings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/boltzgen/resources/config/train/res_type_q_head.yaml"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)

    trainable_substrings = tuple(cfg.get("trainable_parameter_substrings", []))
    trainable_names: list[str] = []
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        param.requires_grad = _matches_any(name, trainable_substrings)
        if param.requires_grad:
            trainable_names.append(name)
            trainable_count += param.numel()

    q_head_names = [
        name for name, _ in model.named_parameters() if "res_type_predictor" in name
    ]

    print("No-data q-head smoke test")
    print("=========================")
    print(f"config:     {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"model.predict_res_type: {getattr(model, 'predict_res_type', None)}")
    print(f"missing keys:           {len(incompatible.missing_keys)}")
    print(f"unexpected keys:        {len(incompatible.unexpected_keys)}")
    print(f"q-head param tensors:   {len(q_head_names)}")
    print(f"trainable params:       {trainable_count:,}/{total_count:,}")

    print("\nq-head parameters:")
    for name in q_head_names:
        print(f"  {name}")

    print("\ntrainable parameters:")
    for name in trainable_names:
        print(f"  {name}")

    if not q_head_names:
        raise SystemExit("FAIL: no res_type_predictor parameters found")
    if not trainable_names:
        raise SystemExit("FAIL: no trainable parameters selected")
    if any("res_type_predictor" not in name for name in trainable_names):
        raise SystemExit("FAIL: non-q-head parameter is trainable")

    print("\nPASS: q head exists and only q-head parameters are trainable.")


if __name__ == "__main__":
    main()

