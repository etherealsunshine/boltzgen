#!/usr/bin/env python3
"""Evaluate a trained BoltzGen q head against real residue identities.

This is the next check after `pilot_train_q_head_multibatch.py`.

It loads:

* a released BoltzGen design checkpoint for the frozen trunk/diffusion model
* a q-head-only checkpoint saved by the multi-batch pilot

Then it runs masked real-data batches and compares q logits against the actual
amino acid labels on design residues.

This does not use inverse-folding p yet. It answers:

    Is the trained BoltzGen q head meaningfully better than random on real,
    masked design residues?

Run from repo root:

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python example/evaluate_q_head_on_real_data.py \\
      --checkpoint /path/to/boltzgen1_diverse.ckpt \\
      --q-head workdir/q_head_pilot/q_head_step500.pt \\
      --batches 64 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F

from boltzgen.task.train.data import TrainingDataModule


AA_TOKEN_OFFSET = 2
AA_TOKEN_COUNT = 20


def _clone_batch(batch: dict) -> dict:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _make_data_config(cfg: OmegaConf, args: argparse.Namespace):
    cfg.data.batch_size = 1
    cfg.data.num_workers = args.num_workers
    cfg.data.samples_per_epoch = max(args.batches * args.max_batch_attempts, 100)
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


def _load_q_head(model: torch.nn.Module, q_head_path: Path) -> None:
    head = model.structure_module.score_model.atom_attention_decoder.res_type_predictor
    state = torch.load(q_head_path, map_location="cpu", weights_only=False)
    head.load_state_dict(state)


def _design_token_count(batch: dict) -> int:
    valid_mask = batch["token_pad_mask"].bool()
    design_mask = batch["design_mask"].bool() & valid_mask
    return int(design_mask.sum().item())


def _next_usable_batch(
    loader_iter,
    data_module: TrainingDataModule,
    device: torch.device,
    *,
    min_design_tokens: int,
    max_attempts: int,
) -> tuple[dict, int]:
    skipped = 0
    for _ in range(max_attempts):
        try:
            batch = next(loader_iter)
            batch = data_module.transfer_batch_to_device(
                batch,
                device,
                dataloader_idx=0,
            )
            batch = _clone_batch(batch)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"skipping failed data batch: {type(exc).__name__}: {exc}")
            continue
        if _design_token_count(batch) >= min_design_tokens:
            return batch, skipped
        skipped += 1

    raise RuntimeError(
        f"Could not find a batch with at least {min_design_tokens} design tokens "
        f"after {max_attempts} attempts. Try lowering --min-design-tokens or "
        "reducing --max-tokens/--max-atoms/--max-seqs."
    )


def _canonical_mask(token: torch.Tensor) -> torch.Tensor:
    return (token >= AA_TOKEN_OFFSET) & (token < AA_TOKEN_OFFSET + AA_TOKEN_COUNT)


@torch.no_grad()
def _extract_selected_logits(
    model: torch.nn.Module,
    batch: dict,
    *,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    true = labels.argmax(dim=-1)
    mask = design_mask & _canonical_mask(true)
    token_index = batch["token_index"][mask].detach().cpu().reshape(-1)

    if not mask.any():
        return (
            torch.empty(0, logits.shape[-1], device=logits.device),
            torch.empty(0, dtype=torch.long, device=logits.device),
            token_index,
        )

    return logits[mask].float(), true[mask], token_index


def _calibration_error(
    confidence: torch.Tensor,
    correct: torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    if confidence.numel() == 0:
        return float("nan")
    total = confidence.numel()
    ece = torch.zeros((), device=confidence.device)
    for bin_idx in range(bins):
        low = bin_idx / bins
        high = (bin_idx + 1) / bins
        in_bin = (confidence > low) & (confidence <= high)
        if not in_bin.any():
            continue
        bin_conf = confidence[in_bin].mean()
        bin_acc = correct[in_bin].float().mean()
        ece = ece + (in_bin.float().sum() / total) * torch.abs(bin_conf - bin_acc)
    return float(ece.item())


def _score_selected_logits(
    selected_logits: torch.Tensor,
    selected_true: torch.Tensor,
    token_index: torch.Tensor,
    *,
    temperature: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if selected_true.numel() == 0:
        return {
            "n": 0.0,
            "loss": float("nan"),
            "top1": float("nan"),
            "top3": float("nan"),
            "top5": float("nan"),
            "entropy": float("nan"),
            "confidence": float("nan"),
            "ece": float("nan"),
            "canonical_mass": float("nan"),
            "noncanonical_top1": float("nan"),
        }, []

    scaled_logits = selected_logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)

    loss = F.cross_entropy(scaled_logits, selected_true)
    topk = torch.topk(probs, k=5, dim=-1)
    pred = topk.indices[:, 0]
    correct = pred == selected_true
    top3 = (topk.indices[:, :3] == selected_true[:, None]).any(dim=-1)
    top5 = (topk.indices == selected_true[:, None]).any(dim=-1)
    confidence = topk.values[:, 0]
    entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
    canonical_mass = probs[:, AA_TOKEN_OFFSET : AA_TOKEN_OFFSET + AA_TOKEN_COUNT].sum(
        dim=-1,
    )
    noncanonical_top1 = ~_canonical_mask(pred)
    ece = _calibration_error(confidence, correct)

    rows = []
    for i in range(selected_true.shape[0]):
        rows.append(
            {
                "token_index": int(token_index[i].item()),
                "true_token": int(selected_true[i].item()),
                "pred_token": int(pred[i].item()),
                "correct": float(correct[i].item()),
                "confidence": float(confidence[i].item()),
                "entropy": float(entropy[i].item()),
                "canonical_mass": float(canonical_mass[i].item()),
                "noncanonical_top1": float(noncanonical_top1[i].float().item()),
                "nll": float(-torch.log(probs[i, selected_true[i]].clamp_min(1e-9)).item()),
            }
        )

    metrics = {
        "n": float(selected_true.numel()),
        "loss": float(loss.item()),
        "top1": float(correct.float().mean().item()),
        "top3": float(top3.float().mean().item()),
        "top5": float(top5.float().mean().item()),
        "entropy": float(entropy.mean().item()),
        "confidence": float(confidence.mean().item()),
        "ece": ece,
        "canonical_mass": float(canonical_mass.mean().item()),
        "noncanonical_top1": float(noncanonical_top1.float().mean().item()),
    }
    return metrics, rows


def _weighted_mean(rows: list[dict[str, float]], key: str) -> float:
    usable = [
        row
        for row in rows
        if row["n"] > 0 and math.isfinite(row["n"]) and math.isfinite(row[key])
    ]
    total = sum(row["n"] for row in usable)
    if total == 0:
        return float("nan")
    return sum(row[key] * row["n"] for row in usable) / total


def _parse_temperatures(value: str | None, fallback: float) -> list[float]:
    if value is None:
        temperatures = [fallback]
    else:
        temperatures = [float(item) for item in value.split(",") if item.strip()]
    if not temperatures:
        raise SystemExit("At least one temperature is required.")
    if any(temp <= 0 for temp in temperatures):
        raise SystemExit("All temperatures must be positive.")
    return temperatures


def _summary_for_temperature(
    batch_metrics: list[dict[str, float]],
    *,
    temperature: float,
) -> dict[str, float | int]:
    rows = [row for row in batch_metrics if row["temperature"] == temperature]
    valid_batches = [
        row
        for row in rows
        if row["n"] > 0 and all(
            math.isfinite(row[key])
            for key in (
                "loss",
                "top1",
                "top3",
                "top5",
                "entropy",
                "confidence",
                "ece",
                "canonical_mass",
                "noncanonical_top1",
            )
        )
    ]
    zero_residue_batches = sum(row["n"] == 0 for row in rows)
    nonfinite_batches = len(rows) - len(valid_batches) - zero_residue_batches
    return {
        "temperature": temperature,
        "valid_batches": len(valid_batches),
        "zero_residue_batches": zero_residue_batches,
        "nonfinite_batches": nonfinite_batches,
        "residues": int(sum(row["n"] for row in valid_batches)),
        "loss": _weighted_mean(rows, "loss"),
        "top1": _weighted_mean(rows, "top1"),
        "top3": _weighted_mean(rows, "top3"),
        "top5": _weighted_mean(rows, "top5"),
        "entropy": _weighted_mean(rows, "entropy"),
        "confidence": _weighted_mean(rows, "confidence"),
        "ece": _weighted_mean(rows, "ece"),
        "canonical_mass": _weighted_mean(rows, "canonical_mass"),
        "noncanonical_top1": _weighted_mean(rows, "noncanonical_top1"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--q-head", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/boltzgen/resources/config/train/res_type_q_head.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("workdir/q_head_eval"))
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperatures",
        help="Comma-separated temperature sweep, e.g. 1,2,4,8.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-atoms", type=int, default=2048)
    parser.add_argument("--max-seqs", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-design-tokens", type=int, default=32)
    parser.add_argument("--max-batch-attempts", type=int, default=100)
    parser.add_argument("--no-msa", action="store_true")
    args = parser.parse_args()
    temperatures = _parse_temperatures(args.temperatures, args.temperature)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(args.config)
    data_cfg = _make_data_config(cfg, args)
    data_module = TrainingDataModule(data_cfg)
    loader_iter = cycle(data_module.train_dataloader())

    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    _load_q_head(model, args.q_head)
    model.to(device)
    model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_batch_path = args.output_dir / "q_head_eval_batches.csv"
    per_residue_path = args.output_dir / "q_head_eval_residues.csv"
    summary_path = args.output_dir / "q_head_eval_summary.json"

    batch_fieldnames = [
        "batch",
        "temperature",
        "n",
        "loss",
        "top1",
        "top3",
        "top5",
        "entropy",
        "confidence",
        "ece",
        "canonical_mass",
        "noncanonical_top1",
        "skipped_batches",
    ]
    residue_fieldnames = [
        "batch",
        "temperature",
        "token_index",
        "true_token",
        "pred_token",
        "correct",
        "confidence",
        "entropy",
        "canonical_mass",
        "noncanonical_top1",
        "nll",
    ]

    print("BoltzGen q-head real-data evaluation")
    print("====================================")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"q_head:          {args.q_head}")
    print(f"device:          {device}")
    print(f"precision:       {args.precision if device.type == 'cuda' else 'fp32'}")
    print(f"temperatures:    {temperatures}")
    print(f"batches:         {args.batches}")
    print(f"min_design:      {args.min_design_tokens}")
    print(f"missing keys:    {len(incompatible.missing_keys)}")
    print(f"unexpected keys: {len(incompatible.unexpected_keys)}")

    batch_metrics: list[dict[str, float]] = []
    total_skipped = 0
    with per_batch_path.open("w", newline="") as batch_handle, per_residue_path.open(
        "w", newline=""
    ) as residue_handle:
        batch_writer = csv.DictWriter(batch_handle, fieldnames=batch_fieldnames)
        residue_writer = csv.DictWriter(residue_handle, fieldnames=residue_fieldnames)
        batch_writer.writeheader()
        residue_writer.writeheader()

        for batch_idx in range(args.batches):
            batch, skipped = _next_usable_batch(
                loader_iter,
                data_module,
                device,
                min_design_tokens=args.min_design_tokens,
                max_attempts=args.max_batch_attempts,
            )
            total_skipped += skipped
            selected_logits, selected_true, token_index = _extract_selected_logits(
                model,
                batch,
                precision=args.precision,
            )
            metrics_by_temp = {}
            for temperature in temperatures:
                metrics, residue_rows = _score_selected_logits(
                    selected_logits,
                    selected_true,
                    token_index,
                    temperature=temperature,
                )
                metrics["temperature"] = temperature
                metrics["skipped_batches"] = float(skipped)
                batch_metrics.append(metrics)
                metrics_by_temp[temperature] = metrics
                batch_writer.writerow({"batch": batch_idx, **metrics})

                for row in residue_rows:
                    residue_writer.writerow(
                        {"batch": batch_idx, "temperature": temperature, **row},
                    )

            if batch_idx == 0 or (batch_idx + 1) % 10 == 0 or batch_idx + 1 == args.batches:
                metrics = metrics_by_temp[temperatures[0]]
                print(
                    f"batch={batch_idx + 1:04d}/{args.batches} "
                    f"T={temperatures[0]:g} n={int(metrics['n'])} "
                    f"loss={metrics['loss']:.4f} "
                    f"top1={metrics['top1']:.3f} top3={metrics['top3']:.3f} "
                    f"top5={metrics['top5']:.3f} skipped={skipped}"
                )

    summaries = {
        str(temperature): _summary_for_temperature(
            batch_metrics,
            temperature=temperature,
        )
        for temperature in temperatures
    }
    primary = summaries[str(temperatures[0])]

    summary = {
        "checkpoint": str(args.checkpoint),
        "q_head": str(args.q_head),
        "requested_batches": args.batches,
        "temperatures": temperatures,
        "primary_temperature": temperatures[0],
        **primary,
        "temperature_summaries": summaries,
        "random_top1_33": 1.0 / 33.0,
        "random_top1_20": 1.0 / 20.0,
        "skipped_attempts": total_skipped,
        "per_batch_csv": str(per_batch_path),
        "per_residue_csv": str(per_residue_path),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSummary")
    print("-------")
    for temperature in temperatures:
        temp_summary = summaries[str(temperature)]
        print(
            f"T={temperature:g} "
            f"valid={temp_summary['valid_batches']}/{summary['requested_batches']} "
            f"res={temp_summary['residues']} loss={temp_summary['loss']:.4f} "
            f"top1={temp_summary['top1']:.4f} "
            f"top3={temp_summary['top3']:.4f} "
            f"top5={temp_summary['top5']:.4f} "
            f"conf={temp_summary['confidence']:.4f} "
            f"ece={temp_summary['ece']:.4f} "
            f"canon={temp_summary['canonical_mass']:.4f} "
            f"noncanon_top1={temp_summary['noncanonical_top1']:.4f}"
        )
    print(f"random top1/33: {summary['random_top1_33']:.4f}")
    print(f"random top1/20: {summary['random_top1_20']:.4f}")
    print(f"skipped:        {summary['skipped_attempts']}")
    print(f"wrote:          {summary_path}")


if __name__ == "__main__":
    main()
