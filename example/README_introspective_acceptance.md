# Introspective Acceptance POC

This is an experimental scaffold for testing an I-DLM-style residue acceptance
policy with BoltzGen.

## What is already available

The released checkpoints expose:

- `p`: inverse-folding anchor logits from `boltzgen1_ifold.ckpt`.
- no `q`: the released design checkpoints were trained with
  `predict_res_type=false`, so they do not expose proposal amino-acid logits.

Confirm with:

```bash
python example/probe_design_q_logits.py --checkpoint /path/to/boltzgen1_diverse.ckpt
python example/probe_design_q_logits.py --checkpoint /path/to/boltzgen1_adherence.ckpt
python example/probe_design_q_logits.py --checkpoint /path/to/boltzgen1_ifold.ckpt
```

## Train a first q head

The config below freezes the released design model and trains only the dormant
`res_type_predictor` proposal head:

```bash
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/res_type_q_head.yaml \
  pretrained=/path/to/boltzgen1_diverse.ckpt \
  output=workdir \
  name=res_type_q_head \
  trainer.devices=1 \
  trainer.max_steps=5000
```

For a quick smoke test, reduce the step count:

```bash
python src/boltzgen/resources/main.py \
  src/boltzgen/resources/config/train/res_type_q_head.yaml \
  pretrained=/path/to/boltzgen1_diverse.ckpt \
  output=workdir \
  name=res_type_q_head_smoke \
  trainer.devices=1 \
  trainer.max_steps=10 \
  data.samples_per_epoch=10 \
  data.num_workers=0
```

The training task prints how many parameters remain trainable. It should be a
small fraction of the full model and should mention `res_type_predictor`.

## Export p/q for acceptance analysis

After you have a design run that writes `res_type_logits` and an inverse-folding
run that writes `inverse_fold_logits`, convert them to JSON:

```bash
python example/export_introspective_pq.py \
  --proposal-npz /path/to/proposal.npz \
  --anchor-npz /path/to/anchor.npz \
  --out /path/to/pq.json
```

Then run the acceptance analysis:

```bash
python example/introspective_acceptance_poc.py --json /path/to/pq.json
```

## Current limitation

This is still analysis-only. It does not mutate residues or rebuild/refold
structures after rejection. That comes after we verify that the trained `q`
head has useful acceptance signal.

