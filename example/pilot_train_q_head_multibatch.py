#!/usr/bin/env python3
"""Small multi-batch pilot for the BoltzGen residue proposal q head.

This is the next rung after `mvp_train_q_head_one_batch.py`.

It freezes a BoltzGen design model, trains only the added
`res_type_predictor` q head over fresh real-data batches, periodically evaluates
on separate fresh batches, and saves the tiny q-head state dict.

This still does not perform p/q acceptance or residue correction. It answers:

    Does the q head generalize beyond one-batch overfit?

Run from repo root:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python example/pilot_train_q_head_multibatch.py \\
      --checkpoint /path/to/boltzgen1_diverse.ckpt \\
      --steps 1000 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from boltzgen.task.train.data import TrainingDataModule


def _clone_batch(batch: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _freeze_except_q_head(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        param.requires_grad = "res_type_predictor" in name
        if param.requires_grad:
            params.append(param)
    return params


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_data_config(cfg: OmegaConf, args: argparse.Namespace):
    cfg.data.batch_size = 1
    cfg.data.num_workers = args.num_workers
    cfg.data.samples_per_epoch = max(args.steps + args.eval_batches + 10, 100)
    cfg.data.max_tokens = args.max_tokens
    cfg.data.max_atoms = args.max_atoms
    cfg.data.max_seqs = min(int(cfg.data.max_seqs), args.max_seqs)
    cfg.data.val_batch_size = 1
    cfg.data.inverse_fold = False
    cfg.data.use_msa = not args.no_msa
    cfg.data.compute_frames = True
    cfg.data.monomer_split = None
    cfg.data.ligand_split = None
    cfg.model.training_args.diffusion_multiplicity = 1
    cfg.model.training_args.diffusion_samples = 1

    return hydra.utils.instantiate(
        OmegaConf.merge(
            {"_target_": "boltzgen.task.train.data.DataConfig"},
            cfg.data,
        )
    )


def _forward_loss(
    model: torch.nn.Module,
    batch: dict,
    *,
    loss_mask: str,
    precision: str,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    device = next(model.parameters()).device
    labels = batch["res_type"].float()
    model_input = model.masker(batch)
    with _autocast_context(device, precision):
        out = model(
            model_input,
            recycling_steps=0,
            multiplicity_diffusion_train=1,
            diffusion_samples=1,
        )
        logits = out["res_type"]
        if logits is None:
            raise RuntimeError("Model returned no res_type logits.")

        valid_mask = batch["token_pad_mask"].bool()
        design_mask = batch["design_mask"].bool() & valid_mask
        mask = design_mask if loss_mask == "design" else valid_mask
        if not mask.any():
            mask = valid_mask

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_true = labels.argmax(dim=-1).reshape(-1)
        flat_mask = mask.reshape(-1)
        loss = F.cross_entropy(flat_logits[flat_mask], flat_true[flat_mask])

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        true = labels.argmax(dim=-1)

        def acc(selected: torch.Tensor) -> float:
            selected = selected.bool()
            if not selected.any():
                return float("nan")
            return (pred[selected] == true[selected]).float().mean().item()

        metrics = {
            "loss": float(loss.detach().item()),
            "acc_all": acc(valid_mask),
            "acc_design": acc(design_mask),
            "tokens": float(valid_mask.sum().item()),
            "design_tokens": float(design_mask.sum().item()),
        }

    return loss, metrics, logits.detach()


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader_iter,
    data_module: TrainingDataModule,
    device: torch.device,
    *,
    batches: int,
    loss_mask: str,
    precision: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    model.train()  # BoltzGen uses training=True path to avoid reverse sampling.
    for _ in range(batches):
        batch = next(loader_iter)
        batch = data_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
        batch = _clone_batch(batch)
        _, metrics, _ = _forward_loss(
            model,
            batch,
            loss_mask=loss_mask,
            precision=precision,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return {f"eval_{key}": value / batches for key, value in totals.items()}


def _save_q_head(model: torch.nn.Module, output_dir: Path, step: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    head = model.structure_module.score_model.atom_attention_decoder.res_type_predictor
    path = output_dir / f"q_head_step{step}.pt"
    torch.save(head.state_dict(), path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/boltzgen/resources/config/train/res_type_q_head.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("workdir/q_head_pilot"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--loss-mask", choices=["design", "valid"], default="design")
    parser.add_argument("--no-msa", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    data_cfg = _make_data_config(cfg, args)
    data_module = TrainingDataModule(data_cfg)
    train_loader = data_module.train_dataloader()
    train_iter = cycle(train_loader)
    eval_iter = cycle(train_loader)

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    model.train()

    q_params = _freeze_except_q_head(model)
    if not q_params:
        raise SystemExit("No res_type_predictor parameters found.")

    optimizer = torch.optim.AdamW(q_params, lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"

    print("BoltzGen q-head multi-batch pilot")
    print("=================================")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"device:          {device}")
    print(f"precision:       {args.precision if device.type == 'cuda' else 'fp32'}")
    print(f"steps:           {args.steps}")
    print(f"loss_mask:       {args.loss_mask}")
    print(f"max_tokens:      {args.max_tokens}")
    print(f"max_atoms:       {args.max_atoms}")
    print(f"missing keys:    {len(incompatible.missing_keys)}")
    print(f"unexpected keys: {len(incompatible.unexpected_keys)}")
    print(f"q-head params:   {sum(p.numel() for p in q_params):,}")
    print(f"metrics:         {metrics_path}")

    fieldnames = [
        "step",
        "split",
        "loss",
        "acc_all",
        "acc_design",
        "tokens",
        "design_tokens",
        "elapsed_sec",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    start = time.time()
    last_logits = None
    for step in range(args.steps + 1):
        batch = next(train_iter)
        batch = data_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
        batch = _clone_batch(batch)

        optimizer.zero_grad(set_to_none=True)
        loss, metrics, last_logits = _forward_loss(
            model,
            batch,
            loss_mask=args.loss_mask,
            precision=args.precision,
        )
        if step < args.steps:
            loss.backward()
            optimizer.step()

        if step == 0 or step == args.steps or step % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"train step={step:05d} loss={metrics['loss']:.4f} "
                f"acc_all={metrics['acc_all']:.3f} "
                f"acc_design={metrics['acc_design']:.3f} "
                f"tokens={int(metrics['tokens'])} "
                f"design={int(metrics['design_tokens'])}"
            )
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "step": step,
                        "split": "train",
                        **metrics,
                        "elapsed_sec": elapsed,
                    }
                )

        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            elapsed = time.time() - start
            eval_metrics = _evaluate(
                model,
                eval_iter,
                data_module,
                device,
                batches=args.eval_batches,
                loss_mask=args.loss_mask,
                precision=args.precision,
            )
            print(
                f"fresh step={step:05d} loss={eval_metrics['eval_loss']:.4f} "
                f"acc_all={eval_metrics['eval_acc_all']:.3f} "
                f"acc_design={eval_metrics['eval_acc_design']:.3f}"
            )
            with metrics_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "step": step,
                        "split": "fresh",
                        "loss": eval_metrics["eval_loss"],
                        "acc_all": eval_metrics["eval_acc_all"],
                        "acc_design": eval_metrics["eval_acc_design"],
                        "tokens": eval_metrics["eval_tokens"],
                        "design_tokens": eval_metrics["eval_design_tokens"],
                        "elapsed_sec": elapsed,
                    }
                )

        if step > 0 and (step % args.save_every == 0 or step == args.steps):
            saved = _save_q_head(model, args.output_dir, step)
            print(f"saved q head: {saved}")

    assert last_logits is not None
    probs = torch.softmax(last_logits, dim=-1)
    print("\nFinal checks")
    print("------------")
    print(f"q_logits shape: {tuple(last_logits.shape)}")
    print(f"q_probs row-sum mean: {probs.sum(dim=-1).mean().item():.4f}")
    print(f"latest q-head checkpoint: {args.output_dir / f'q_head_step{args.steps}.pt'}")


if __name__ == "__main__":
    main()
