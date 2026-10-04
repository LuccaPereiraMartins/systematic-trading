# Post-training experiments

The first experiment is implemented in `train.py`: supervised adaptation of Laya's decision head with document attention. It uses the frozen splits in `data/splits`: 8,000 train, 1,000 validation, 1,000 test. Training reads train and validation only. Choose checkpoints, context handling and hyperparameters on validation, then compare the selected model with the recorded base Laya result on test.

## Local hardware and approach

The current machine has an RTX 3060 with 12 GB VRAM and about 16 GB system RAM. Laya's English checkpoint has 421M parameters, including a ModernBERT encoder and a decision head ([model card](https://huggingface.co/convaiinnovations/laya)). The first full head-adaptation run completed locally; cloud compute was not needed.

1. **Supervised head adaptation:** freeze the encoder and train the existing decision head plus an 80-parameter window attention pooler with document classification cross-entropy.
2. **LoRA plus the decision head:** adapt selected encoder layers with low-rank updates while training the head. LoRA controls which weights change; cross-entropy still supplies the supervised objective. It reduces trainable parameters and optimizer memory ([PEFT documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)). Exact modules and export/reload compatibility need checking against Laya's custom architecture.
3. **RLCD experiment:** compare supervised adaptation with Laya's proper-scoring-rule reward and policy-gradient recipe, starting from the same base or explicitly recording an SFT initialization. The [upstream training notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) is a reference, not a drop-in finance pipeline.

This experiment uses mixed precision, one document per backward pass and gradient accumulation. Measure LoRA memory separately before deciding whether it needs cloud compute.

## Run the supervised experiment

```powershell
# Small mechanics check; separate artifacts from the real experiment:
python decision_models/train.py --limit 48 --output decision_models/training_runs/smoke
# One complete epoch, with the full separate validation set:
python decision_models/train.py
# Resume the same run after an interruption:
python decision_models/train.py --resume
# Evaluate the validation-selected checkpoint on the held-out test:
python decision_models/benchmark.py --models head --device cuda --splits decision_models/data/splits --checkpoint decision_models/training_runs/head/best.pt
```

Defaults are visible in `train.py`: seed 42, AdamW at 1e-4, one epoch, gradient accumulation over 8 documents, 8 windows per forward, bf16 on CUDA and activation checkpointing in the decision head. The encoder and unused action head are frozen; 26,248,273 parameters remain trainable. The default uses unweighted cross-entropy and retains every token window. `--epochs` can extend a resumed run; `--output` separates experiments. No further packages are needed beyond the benchmark dependencies.

The 48-document smoke run completed, saved and reloaded weights, and resumed without repeating completed training. An inline check confirmed finite gradients reaching the scorer and pooler but none reaching the encoder. The largest training filing (144 windows) fitted at about 2.42 GiB of allocated CUDA memory during backward. These checks establish mechanics and memory feasibility, not model quality.

Each ignored run directory contains `config.json`, `history.json`, `last.pt`, `best.pt` and versioned source snapshots under `source/`. Configuration records base revision, question/label ordering, seed, objective, batch/context settings, source-file fingerprints, Git revision, dependency versions, split checksums and GPU. History records per-epoch training loss and validation metrics, elapsed seconds and peak allocated VRAM. Checkpoints contain trainable weights, optimizer state, epoch/document offset and RNG states; frozen encoder weights reload from the pinned base checkpoint. Progress is checkpointed every 200 documents at optimizer-step boundaries. Resuming rejects changed training settings, runtime versions or splits, except for extending epochs. Source/Git/GPU changes are recorded in resume history; original source snapshots are retained. For exact reproduction, use the matching snapshot and recorded versions rather than assuming later code edits preserve behavior.

`best.pt` is selected by validation macro F1, including the initial uniform-pooling baseline at epoch 0. If adaptation does not improve that baseline, the selected checkpoint can remain epoch 0. The benchmark loads the saved weights and uses the same windowing and pooling as training. It records the checkpoint fingerprint and training configuration. Artifacts remain local; transfer the run folder alongside the dataset ZIP to share an experiment.

Our labels describe whole documents. Each document is tokenized without truncation and split into 312-token windows with 50% overlap (the pinned checkpoint has 512 total tokens and a 192-token question budget). Laya's existing renderer inserts the same question/option markers in each window; the script verifies no input tokens were dropped. The existing head scores each window. A small MLP consumes the three per-window log-probabilities and learns attention weights across windows; their weighted raw logits supply one document prediction and one cross-entropy loss. Windows receive no individual labels. The pooler starts with uniform weights, and the frozen encoder stays in evaluation mode while the head trains with its existing dropout.

This uses the document-label idea of [multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html). Attention weights identify which windows influenced this architecture, not proven causal explanations. Inference uses the same learned pooler rather than base Laya's most-confident-window selection. Any measured change therefore includes both head adaptation and pooling; the initial pooled validation result provides a check on the aggregation change.

Human annotations take precedence. Luna labels are provisional targets; their uncertainty scalar is not a probability distribution. Use validation for calibration and report agreement, macro F1, review-worthy recall, inference latency, and training compute/cost. Add probability metrics once document-level probabilities are validated. Temperature fitting must not reuse training labels or test labels.

## First supervised run — 2026-10-04

One epoch covered all 8,000 training documents (77,740 overlapping windows), with 1,000 optimizer steps and 1,000 separate validation documents (9,375 windows). It took 44.0 minutes including token preparation and both validation passes, with 2.72 GiB peak allocated CUDA memory. API cost was $0; electricity and hardware are excluded. Inputs were the stored raw bodies, without cleaning. Mean training cross-entropy was 0.762.

| Validation checkpoint | Agreement with Luna | Macro F1 | Cross-entropy |
| --- | ---: | ---: | ---: |
| Initial head + uniform pooling (epoch 0) | 57.1% | 0.345 | 0.874 |
| Adapted head + learned attention (epoch 1) | 66.0% | 0.352 | 0.707 |

Epoch 1 was selected by validation macro F1. It predicted 964 review-worthy, 4 routine and 32 unclear documents; review-worthy recall was 97.9%, but routine recall was only 0.8%. The higher agreement mostly reflects movement toward the majority class. This first run establishes a working training pipeline, with little improvement in balanced classification. A next experiment should compare class-balanced loss and learning rates on validation before proceeding to LoRA. Held-out results are recorded in [benchmarks.md](benchmarks.md).

The split removes exact body overlap and reserves the 101 bodies from the earlier pilot for training. Related issuer documents and near-duplicate templates can still cross boundaries; this random stratified split does not measure future-period or unseen-company generalization. Add those evaluations later. Future human label revisions should produce a new dataset/split version.
