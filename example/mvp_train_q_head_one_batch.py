#!/usr/bin/env python3
"""Smallest real-data MVP for the BoltzGen introspective-acceptance q head.

This script validates the missing piece for I-DLM-style acceptance:

    q = BoltzGen design proposal logits over amino-acid identity

The released design checkpoints do not include q logits, but the architecture
can add a `res_type_predictor` head. This MVP freezes the released design model,
trains only that q head on one real BoltzGen training batch, and reports whether
the head can overfit residue identities above random.

It intentionally does NOT run full reverse diffusion sampling and does NOT need
inverse folding. The acceptance math and inverse-folding p-head are validated
elsewhere; this script answers the most immediate question:

    Can the added q head learn real residue identity signal from BoltzGen states?

Expected after a successful short run:

    step=... loss decreases
    acc_all / acc_design improves above random
    q_logits shape is [B, N, 33]

Run from repo root after downloading training_data:

    python example/mvp_train_q_head_one_batch.py \
      --checkpoint /path/to/boltzgen1_diverse.ckpt \
      --steps 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from boltzgen.model.models.boltz import Boltz
from boltzgen.task.train.data import TrainingDataModule


def _freeze_except_q_head(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        param.requires_grad = "res_type_predictor" in name
        if param.requires_grad:
            params.append(param)
    return params


def _accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    if not mask.any():
        return float("nan")
    pred = logits.argmax(dim=-1)
    true = target.argmax(dim=-1)
    return (pred[mask] == true[mask]).float().mean().item()


def _clone_batch(batch: dict) -> dict:
    """Clone tensor leaves because BoltzGen forward mutates some feature tensors."""

    cloned = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/boltzgen/resources/config/train/res_type_q_head.yaml"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)

    # Keep the data footprint small for the MVP.
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.samples_per_epoch = 1
    cfg.data.max_tokens = min(int(cfg.data.max_tokens), 256)
    cfg.data.max_atoms = min(int(cfg.data.max_atoms), 2048)
    cfg.data.val_batch_size = 1
    cfg.data.inverse_fold = False
    cfg.data.use_msa = True
    cfg.data.compute_frames = True
    cfg.model.training_args.diffusion_multiplicity = 1
    cfg.model.training_args.diffusion_samples = 1

    data_cfg = hydra.utils.instantiate(
        OmegaConf.merge(
            {"_target_": "boltzgen.task.train.data.DataConfig"},
            cfg.data,
        )
    )
    data_module = TrainingDataModule(data_cfg)
    loader = data_module.train_dataloader()
    base_batch = next(iter(loader))
    base_batch = data_module.transfer_batch_to_device(
        base_batch,
        device,
        dataloader_idx=0,
    )

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.train()

    q_params = _freeze_except_q_head(model)
    if not q_params:
        raise SystemExit("No res_type_predictor parameters found.")

    optimizer = torch.optim.AdamW(q_params, lr=args.lr)

    print("BoltzGen q-head one-batch MVP")
    print("=============================")
    print(f"checkpoint:       {args.checkpoint}")
    print(f"device:           {device}")
    print(f"missing keys:     {len(incompatible.missing_keys)}")
    print(f"unexpected keys:  {len(incompatible.unexpected_keys)}")
    print(f"q-head params:    {sum(p.numel() for p in q_params):,}")
    print(f"tokens in batch:  {int(base_batch['token_pad_mask'].sum().item())}")
    print(f"design tokens:    {int(base_batch['design_mask'].bool().sum().item())}")

    last_logits = None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = _clone_batch(base_batch)
        out = model(
            batch,
            recycling_steps=0,
            multiplicity_diffusion_train=1,
            diffusion_samples=1,
        )
        logits = out["res_type"]
        if logits is None:
            raise SystemExit("Model returned no res_type logits.")

        target = batch["res_type"].float()
        valid_mask = batch["token_pad_mask"].bool()
        design_mask = (batch["design_mask"].bool() & valid_mask)

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_true = target.argmax(dim=-1).reshape(-1)
        flat_mask = valid_mask.reshape(-1)
        loss = F.cross_entropy(flat_logits[flat_mask], flat_true[flat_mask])

        if step < args.steps:
            loss.backward()
            optimizer.step()

        last_logits = logits.detach()

        if step == 0 or step == args.steps or step % max(1, args.steps // 10) == 0:
            acc_all = _accuracy(logits.detach(), target, valid_mask)
            acc_design = _accuracy(logits.detach(), target, design_mask)
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"acc_all={acc_all:.3f} acc_design={acc_design:.3f}"
            )

    assert last_logits is not None
    probs = torch.softmax(last_logits, dim=-1)
    print("\nFinal checks")
    print("------------")
    print(f"q_logits shape: {tuple(last_logits.shape)}")
    print(f"q_probs row-sum mean: {probs.sum(dim=-1).mean().item():.4f}")
    print("PASS if loss decreased and accuracy rose above random (~0.03 for 33 classes).")


if __name__ == "__main__":
    main()
