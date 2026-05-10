# Introspective Acceptance for BoltzGen: MVP Research Log

## 1. Goal

We are exploring whether the I-DLM introspective acceptance idea can be adapted
to all-atom protein design with BoltzGen.

In I-DLM-style decoding, a model produces a proposal distribution `q` and then
checks the sampled token against an introspective anchor distribution `p`:

```text
candidate x ~ q
accept x with probability min(1, p(x) / q(x))
if rejected, resample x ~ normalize(max(0, p - q))
```

For BoltzGen, the analogous residue-level version is:

```text
q_i = BoltzGen design proposal distribution over amino acid identity at residue i
p_i = anchor/verifier distribution over amino acid identity at residue i

accept proposed AA_i with min(1, p_i(AA_i) / q_i(AA_i))
```

The eventual hypothesis is:

```text
Residues that are locally inconsistent under an anchor verifier can be detected
and corrected before or during downstream refolding/filtering.
```

This is not yet a full all-atom correction pipeline. The current MVP is focused
on establishing whether the missing probability objects `p` and `q` can exist
in BoltzGen.

## 2. Why This Is Nontrivial for BoltzGen

I-DLM is discrete-language-model machinery. BoltzGen is all-atom and
bidirectional/context-dependent.

This creates two mismatches:

1. BoltzGen does not naturally expose a causal AR anchor like I-DLM.
2. The released BoltzGen design checkpoints do not expose residue proposal
   logits `q`.

So we reframed the idea:

```text
I-DLM causal anchor
    -> protein structural anchor / inverse-folding verifier

I-DLM token proposal q
    -> BoltzGen residue-type proposal head q
```

At this stage we restrict the acceptance variable to amino acid identity:

```text
AA_i in 33 BoltzGen token classes
```

Rotamers, sidechain coordinates, and all-atom refinement are later extensions.

## 3. What We Found in Released Checkpoints

We added:

```text
example/probe_design_q_logits.py
```

This probes checkpoints for residue classifier heads.

Observed:

```text
boltzgen1_diverse.ckpt
    hyper_parameters.predict_res_type: False
    hyper_parameters.inverse_fold:      False
    design q head weights found:        False

boltzgen1_adherence.ckpt
    hyper_parameters.predict_res_type: False
    hyper_parameters.inverse_fold:      False
    design q head weights found:        False

boltzgen1_ifold.ckpt
    hyper_parameters.predict_res_type: True
    hyper_parameters.inverse_fold:      True
    inverse-fold p head weights found:  True
    structure_module.predictor.weight:  (33, 128)
```

Conclusion:

```text
p anchor logits exist in inverse folding.
q proposal logits do not exist in released design checkpoints.
```

Therefore Option A became:

```text
Add a small residue-type q head to the design model and train only that head.
```

## 4. Acceptance Math POC

We added:

```text
src/boltzgen/experimental/introspective_acceptance.py
example/introspective_acceptance_poc.py
tests/test_introspective_acceptance.py
```

This validates the exact categorical acceptance rule:

```text
candidate z ~ q
accept z with min(1, p(z) / q(z))
otherwise resample z ~ normalize(max(0, p - q))
```

Toy result:

```text
Exact expected acceptance rate: 0.738
Monte Carlo acceptance rate:    0.740
L1(exact final, anchor p):      0.000000
L1(MC final, anchor p):         ~0.006
```

Interpretation:

```text
The accept/correct kernel is implemented correctly.
If q over-proposes a residue that p dislikes, rejection shifts samples toward p.
If q and p are aligned, acceptance is high.
```

## 5. q-Head Architecture Smoke Tests

The design architecture already has a dormant `predict_res_type` pathway. We
validated that enabling it adds:

```text
structure_module.score_model.atom_attention_decoder.res_type_predictor.weight
structure_module.score_model.atom_attention_decoder.res_type_predictor.bias
```

No-data smoke result:

```text
model.predict_res_type: True
q-head param tensors:   2
trainable params:       4,257 / 137,303,715

q-head parameters:
  structure_module.score_model.atom_attention_decoder.res_type_predictor.weight
  structure_module.score_model.atom_attention_decoder.res_type_predictor.bias

PASS: q head exists and only q-head parameters are trainable.
```

Tiny head-only check:

```text
Linear(in_features=128, out_features=33, bias=True)
head weight: (33, 128)
logits: (4, 33)
probs row sums: tensor([1., 1., 1., 1.])
out_dim: 33
```

Conclusion:

```text
The q head can be added cleanly and has the expected shape.
```

## 6. Real-Data One-Batch MVP

We added:

```text
example/mvp_train_q_head_one_batch.py
```

Purpose:

```text
Use real BoltzGen training data.
Load released design checkpoint.
Enable the q head.
Freeze everything except q head.
Train q head on one real batch.
Check whether q can learn residue identity signal.
```

This is the smallest meaningful real-data validation of the missing `q` object.

The script intentionally does not:

```text
run full reverse diffusion sampling
use inverse folding
correct residues
refold structures
claim generalization
```

It only asks:

```text
Can BoltzGen internal design states support a learned residue proposal q?
```

## 7. Training Data Requirement

The training dataloader requires:

```text
training_data/targets/manifest.json
training_data/targets/structures/
training_data/targets/records/
training_data/msa/
training_data/mols/
```

Upstream data sizes are large:

```text
targets.zip: ~99 GB
msa.zip:     ~70 GB
unzipped:    larger
```

This is why no-data and tiny tests were useful first. The one-batch MVP requires
the real training data.

## 8. Real-Data MVP Results

### 1-step / 20-step run

Observed:

```text
q-head params:    4,257
tokens in batch:  233-255
design tokens:    192-211
q_logits shape:   (1, 256, 33)
q_probs row-sum mean: 1.0000
```

One 20-step run:

```text
step=0000 loss=93.4963 acc_all=0.004 acc_design=0.000
step=0002 loss=72.7958 acc_all=0.017 acc_design=0.014
step=0004 loss=56.2898 acc_all=0.064 acc_design=0.066
...
step=0020 loss=13.9808 acc_all=0.064 acc_design=0.062
```

Interpretation:

```text
Real training data loads.
Forward path produces q logits.
Loss computes.
Repeated optimizer steps run.
Loss drops substantially.
Accuracy rises above random baseline.
```

Random baseline is approximately:

```text
1 / 33 ~= 0.030
```

### 100-step run

Observed:

```text
step=0010 loss=23.3940 acc_all=0.055 acc_design=0.077
step=0020 loss=14.7637 acc_all=0.071 acc_design=0.058
step=0030 loss=10.2059 acc_all=0.038 acc_design=0.058
step=0040 loss=7.0641  acc_all=0.050 acc_design=0.077
step=0050 loss=5.0760  acc_all=0.101 acc_design=0.038
step=0060 loss=5.3778  acc_all=0.059 acc_design=0.115
step=0070 loss=4.3070  acc_all=0.080 acc_design=0.058
step=0080 loss=4.3831  acc_all=0.097 acc_design=0.154
step=0090 loss=4.4026  acc_all=0.105 acc_design=0.096
step=0100 loss=4.0838  acc_all=0.101 acc_design=0.115

Final checks:
q_logits shape: (1, 256, 33)
q_probs row-sum mean: 1.0000
```

Interpretation:

```text
The q head can overfit signal from real BoltzGen states.
Loss falls strongly.
Accuracy is noisy but consistently above random in later steps.
This validates architectural feasibility of q.
```

Important caveat:

```text
This is one-batch overfitting, not generalization.
```

## 9. Bugs/Issues Encountered and Fixed

### Wrong package: `hydra` vs `hydra-core`

Initial environment tried to install old `hydra==2.5`, causing:

```text
longintrepr.h: No such file or directory
```

Fix:

```bash
pip uninstall -y hydra
pip install hydra-core
```

### Missing q-head in released design checkpoint

Confirmed by checkpoint probe. This is expected. We add the head with:

```text
predict_res_type=True
strict=False
```

### Full reverse diffusion SVD failure

Sampling path hit:

```text
torch._C._LinAlgError: linalg.svd failed to converge
```

We avoided this for the MVP by using the training/noising forward path instead
of full reverse diffusion sampling.

### OOM / killed process

Full trunk forward on large example YAMLs can OOM. The MVP reduces:

```text
batch_size = 1
diffusion_multiplicity = 1
diffusion_samples = 1
max_tokens <= 256
max_atoms <= 2048
only q head trainable
```

For CUDA, use:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Batch mutation between steps

BoltzGen forward mutates feature tensors such as `coords`. Reusing the same
batch caused an assertion failure on step 1.

Fix:

```text
clone tensor leaves before each step
```

## 10. What This MVP Proves

The MVP proves:

```text
1. The acceptance kernel works.
2. The inverse-folding checkpoint provides anchor p logits.
3. Released design checkpoints do not provide q logits.
4. A q head can be added cleanly.
5. The q head can train on real BoltzGen states.
6. Real q logits have valid shape: (B, N, 33).
7. One-batch overfit improves loss and accuracy above random.
```

This is enough to justify the next experiment.

## 11. What It Does Not Prove Yet

It does not prove:

```text
q generalizes across proteins
q is calibrated as a proposal distribution
p and q are comparable distributions
p/q acceptance improves design quality
residue corrections preserve all-atom geometry
sidechain/rotamer issues are solved
```

The current `q` head is still only amino-acid identity.

## 12. Next Step: Small Multi-Batch Pilot

Move from one-batch overfit to small multi-batch training:

```text
train q head for 1k-5k steps
use fresh batches
save checkpoint
evaluate held-out batches
track res_type_loss and res_type_acc
```

Success criteria:

```text
held-out residue accuracy above random
loss meaningfully below random/class-frequency baseline
probabilities not collapsed to a few residue types
design-token accuracy improves
```

This answers:

```text
Can q generalize enough to be useful for p/q acceptance?
```

## 13. Next Step After q Generalization

Once q is usable:

1. Run design model with trained q head and export:

```text
res_type_logits -> q
```

2. Run inverse folding and export:

```text
inverse_fold_logits -> p
```

3. Convert to p/q JSON:

```bash
python example/export_introspective_pq.py \
  --proposal-npz path/to/proposal.npz \
  --anchor-npz path/to/anchor.npz \
  --out path/to/pq.json
```

4. Analyze acceptance:

```bash
python example/introspective_acceptance_poc.py --json path/to/pq.json
```

Look for:

```text
acceptance rates by residue
low-acceptance residues at interfaces/cores
correlation with failed refolding/filtering metrics
```

## 14. Eventual Full Pipeline

If p/q analysis is meaningful:

```text
BoltzGen proposes design
q head emits proposal distribution
inverse-folding/anchor emits p
accept/reject residue identities
if rejected, resample AA from normalize(max(0, p - q))
rebuild/refold structure
compare downstream design metrics
```

Possible evaluation metrics:

```text
BoltzGen filtering score
refolding RMSD
interface confidence
salt bridges / hydrogen bonds
delta SASA
sequence diversity
liability filters
acceptance rate distribution
```

## 15. Current Bottom Line

The current state is promising but early:

```text
The missing q object is not present in released BoltzGen checkpoints.
But the model can host a q head.
That q head trains on real BoltzGen data and learns residue signal.
```

The next decisive experiment is generalization:

```text
Can a frozen-trunk q head trained across batches provide useful proposal
probabilities for p/q introspective acceptance?
```

