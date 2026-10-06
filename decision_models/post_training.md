# Post-training experiments

`train.py` trains whole-document attention heads and LoRA adapters for Laya, FinBERT and BGE. `encoders.py` supplies the FinBERT/BGE variants; `tune.py` fits linear baselines and chooses decision offsets. All use the frozen splits in `data/splits`: 8,000 train, 1,000 validation, 1,000 test. Training and tuning read train/validation only. Freeze the selected candidates before evaluating test with `benchmark.py --saved`.

## Local hardware and approach

The current machine has an RTX 3060 with 12 GB VRAM and about 16 GB system RAM. Laya's English checkpoint has 421M parameters, including a ModernBERT encoder and a decision head ([model card](https://huggingface.co/convaiinnovations/laya)). The first full head-adaptation run completed locally; cloud compute was not needed.

1. **Supervised head adaptation:** freeze the encoder and train the existing decision head plus an 80-parameter window attention pooler with document classification cross-entropy.
2. **LoRA plus the decision head:** adapt selected encoder layers while training the head. Laya targets ModernBERT's `Wqkv`; FinBERT/BGE target `query` and `value`. Rank 8 uses alpha 16 and adapter dropout 0.05. LoRA controls which weights change; document cross-entropy still supplies the supervised objective ([PEFT documentation](https://huggingface.co/docs/peft/main/en/package_reference/lora)).
3. **RLCD experiment:** compare supervised adaptation with Laya's proper-scoring-rule reward and policy-gradient recipe, starting from the same base or explicitly recording an SFT initialization. The [upstream training notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) is a reference, not a drop-in finance pipeline.

These experiments use mixed precision, one document per backward pass and gradient accumulation. Measure LoRA memory separately before deciding whether it needs cloud compute. Install the adapter dependency into global Python with `uv pip install --system peft==0.21.2`.

## Run the supervised experiment

```powershell
# Small mechanics check; separate artifacts from the real experiment:
python decision_models/train.py --limit 48 --output decision_models/training_runs/smoke
# One complete epoch, with the full separate validation set:
python decision_models/train.py --output decision_models/training_runs/head-tuned
# Resume the same run after an interruption:
python decision_models/train.py --resume --output decision_models/training_runs/head-tuned
# After freezing selection, evaluate the checkpoint and its decision offsets:
python decision_models/benchmark.py --saved decision_models/training_runs/head-tuned --device cuda --splits decision_models/data/splits
```

Defaults are visible in `train.py`: seed 42, AdamW at 1e-4, one epoch, gradient accumulation over 8 documents, 8 windows per forward, bf16 on CUDA and activation checkpointing in the decision head. Default Laya head adaptation freezes the encoder and unused action head; 26,248,273 parameters remain trainable. It uses unweighted cross-entropy and retains every token window. `--epochs` can extend a resumed run; `--output` separates experiments.

The 48-document smoke run completed, saved and reloaded weights, and resumed without repeating completed training. An inline check confirmed finite gradients reaching the scorer and pooler but none reaching the encoder. The largest training filing (144 windows) fitted at about 2.42 GiB of allocated CUDA memory during backward. These checks establish mechanics and memory feasibility, not model quality.

Each ignored run directory contains `config.json`, `history.json`, `last.pt`, `best.pt` and versioned source snapshots under `source/`. Configuration records base revision, question/label ordering, seed, objective, batch/context settings, source-file fingerprints, Git revision, dependency versions, split checksums and GPU. History records per-epoch training loss and validation metrics, elapsed seconds and peak allocated VRAM. Checkpoints contain trainable weights, optimizer state, epoch/document offset and RNG states; frozen encoder weights reload from the pinned base checkpoint. Progress is checkpointed every 200 documents at optimizer-step boundaries. Resuming rejects changed training settings, runtime versions or splits, except for extending epochs. Source/Git/GPU changes are recorded in resume history; original source snapshots are retained. For exact reproduction, use the matching snapshot and recorded versions rather than assuming later code edits preserve behavior.

`best.pt` is selected by validation macro F1 after decision-offset tuning, including the initial uniform-pooling baseline at epoch 0. If adaptation does not improve that baseline, the selected checkpoint can remain epoch 0. The first historical run below used raw macro F1 (`--selection raw`). The benchmark loads the saved weights and uses the same windowing and pooling as training. It records the checkpoint fingerprint and training configuration. Artifacts remain local; transfer the run folder alongside the dataset ZIP to share an experiment.

Our labels describe whole documents. Each document is tokenized without truncation and split into 312-token windows with 50% overlap (the pinned checkpoint has 512 total tokens and a 192-token question budget). Laya's existing renderer inserts the same question/option markers in each window; the script verifies no input tokens were dropped. The existing head scores each window. A small MLP consumes the three per-window log-probabilities and learns attention weights across windows; their weighted raw logits supply one document prediction and one cross-entropy loss. Windows receive no individual labels. The pooler starts with uniform weights, and the frozen encoder stays in evaluation mode while the head trains with its existing dropout.

This uses the document-label idea of [multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html). Attention weights identify which windows influenced this architecture, not proven causal explanations. Inference uses the same learned pooler rather than base Laya's most-confident-window selection. Any measured change therefore includes both head adaptation and pooling; the initial pooled validation result provides a check on the aggregation change.

Human annotations take precedence. Luna labels are provisional targets; their uncertainty scalar is not a probability distribution. Use validation for calibration and report agreement, macro F1, review-worthy recall, inference latency, and training compute/cost. Add probability metrics once document-level probabilities are validated. Temperature fitting must not reuse training labels or test labels.

## First supervised run — 2026-10-04

One epoch covered all 8,000 training documents (77,740 overlapping windows), with 1,000 optimizer steps and 1,000 separate validation documents (9,375 windows). It took 44.0 minutes including token preparation and both validation passes, with 2.72 GiB peak allocated CUDA memory. API cost was $0; electricity and hardware are excluded. Inputs were the stored raw bodies, without cleaning. Mean training cross-entropy was 0.762.

| Validation checkpoint | Agreement with Luna | Macro F1 | Cross-entropy |
| --- | ---: | ---: | ---: |
| Initial head + uniform pooling (epoch 0) | 57.1% | 0.345 | 0.874 |
| Adapted head + learned attention (epoch 1) | 66.0% | 0.352 | 0.707 |

Epoch 1 was selected by validation macro F1. It predicted 964 review-worthy, 4 routine and 32 unclear documents; review-worthy recall was 97.9%, but routine recall was only 0.8%. The higher agreement mostly reflects movement toward the majority class. This first run establishes a working training pipeline, with little improvement in balanced classification. The validation experiments below compare class weighting, decision tuning and LoRA. Held-out results are recorded in [benchmarks.md](benchmarks.md).

The split removes exact body overlap and reserves the 101 bodies from the earlier pilot for training. Related issuer documents and near-duplicate templates can still cross boundaries; this random stratified split does not measure future-period or unseen-company generalization. Add those evaluations later. Future human label revisions should produce a new dataset/split version.

## Validation search protocol

Class weights use training counts only: `(N / (3 * class_count)) ** power`, normalized to have training mean 1. Powers 0, 0.5 and 1 mean unweighted, square-root and full inverse-frequency weighting. Multiply each document's scalar cross-entropy by its weight before averaging across eight documents. Passing class weights to a one-document weighted mean would cancel the weight.

FinBERT and BGE retain every raw 510-token window, with two special tokens added. FinBERT uses masked mean token embeddings; BGE uses the CLS embedding. A small attention MLP pools window embeddings before a three-label classifier. Frozen-head runs cache deterministic window embeddings; LoRA runs recompute them and backpropagate through the adapters. Each document receives one loss; its individual windows receive no labels.

Linear baselines use word/character TF-IDF or the mean of all frozen encoder windows followed by logistic regression. Vocabulary and scaling are fitted on train only. Choose regularization, class weighting and two relative log-probability offsets jointly on validation. The offset grid spans -3 to 3 in 0.25 steps; review-worthy stays at zero. Neural checkpoints now use macro F1 after the same decision tuning (`--selection tuned`, the default) and save `decision.json` automatically after a full run. `--selection raw` reproduces the earlier selection rule and resumes older raw-selected runs; use `tune.py --offsets-only` afterward for those. Offsets change decisions and do not calibrate probabilities. An optional two-model blend also selects its weight and offsets on validation.

Feature caches include raw body fingerprints, model revisions, runtime versions, device and encoder source. Changes create a separate cache rather than silently mixing experiments. CUDA encoder features use bf16 forward passes, with pooling and stored vectors in fp32.

```powershell
# A reproducible linear search, including decision offsets:
python decision_models/tune.py --model tfidf --output decision_models/training_runs/tfidf-joint
# Class-weighted Laya head adaptation:
python decision_models/train.py --epochs 2 --learning-rate 0.00003 --class-weight-power 1 --output decision_models/training_runs/weighted-head-3e-5
# FinBERT document attention + LoRA:
python decision_models/train.py --model finbert --adaptation lora --lora-rank 8 --epochs 3 --learning-rate 0.001 --encoder-learning-rate 0.0001 --class-weight-power 1 --output decision_models/training_runs/finbert-lora
# Decision tuning after training, using validation predictions only:
python decision_models/tune.py --offsets-only --output decision_models/training_runs/finbert-lora
# Only after freezing candidate selection:
python decision_models/benchmark.py --saved decision_models/training_runs/finbert-lora --device cuda --splits decision_models/data/splits
```

Reruns of training require `--resume` or a new output directory. Full trials and source snapshots stay in ignored run folders. Keep the selected configuration and topline results in Markdown. The test split was already used for the first base/head report above; this new search does not use those test predictions for tuning. Later research needs a fresh temporal/issuer-disjoint holdout and human-reviewed labels.

## Current validation results — 2026-10-06

All scores below use the same 1,000 validation documents. Hyperparameters, epochs, class weights and decision offsets are selected here; these are not final test scores. Laya LoRA stopped after one complete epoch at the user's request. Failed variants remain in the ignored experiment log.

| Approach | Raw macro F1 | Tuned macro F1 |
| --- | ---: | ---: |
| Original unweighted Laya head, epoch 1 | 0.352 | 0.534 |
| Weighted Laya head, LR 3e-5, epoch 2 | 0.536 | 0.549 |
| Weighted Laya head + LoRA, epoch 1 | 0.617 | 0.644 |
| Word TF-IDF + logistic regression | 0.631 | 0.685 |
| Character TF-IDF + logistic regression | 0.671 | 0.689 |
| Frozen FinBERT mean windows + logistic regression | 0.560 | 0.661 |
| Frozen FinBERT attention head | 0.631 | 0.650 |
| FinBERT attention + LoRA, epoch 3 | 0.572 | 0.695 |
| Frozen BGE mean windows + logistic regression | 0.508 | 0.637 |
| BGE attention + LoRA, epoch 2 | 0.657 | 0.677 |
| Character TF-IDF / BGE LoRA, 50/50 blend | 0.691 | 0.712 |

The controlled Laya comparison uses the same LR 1e-4 and one epoch: inverse-frequency weights improve raw F1 from 0.352 to 0.523, but tuned F1 only from 0.534 to 0.540. Decision tuning explains most of that apparent improvement. The lower-rate, two-epoch trial is a separate comparison.

Earlier neural trials selected checkpoints by raw F1. FinBERT epoch 2 won that rule, but epoch 3 wins after decision tuning; both weight snapshots were preserved and compared on validation. The current default selects tuned F1 at each epoch directly. BGE selects epoch 2 under either rule. Frozen mean-window baselines and adapted attention models also differ in pooling, so their differences do not isolate LoRA alone.

FinBERT LoRA trains 323,395 parameters and BGE LoRA 161,731, with observed peak allocated VRAM of 1.45 and 0.65 GiB respectively. The Laya LoRA longest-document backward check passed at 144 windows and 8.78 GiB, with finite loss, nonzero adapter gradients and no frozen-parameter gradients. No cloud compute or API calls were used. Some earlier training runs shared the GPU and crossed a roughly six-hour computer disconnect/suspend; their logged elapsed seconds are not isolated active training durations.

Fresh-run commands for the validation-selected blend:

```powershell
python decision_models/tune.py --model char --regularization .001 .01 .1 1 10 30 100 --output decision_models/training_runs/char-wide
python decision_models/train.py --model bge --adaptation lora --lora-rank 8 --epochs 2 --learning-rate .001 --encoder-learning-rate .0001 --class-weight-power 1 --output decision_models/training_runs/bge-lora
python decision_models/tune.py --blend decision_models/training_runs/char-wide decision_models/training_runs/bge-lora --output decision_models/training_runs/blend-char-bge-lora
# After all candidate selection is frozen:
python decision_models/benchmark.py --saved decision_models/training_runs/char-wide decision_models/training_runs/bge-lora decision_models/training_runs/blend-char-bge-lora --device cuda --splits decision_models/data/splits
```

The Laya trial uses head LR 3e-5, encoder LR 1e-4, inverse-frequency weights and rank 8. One complete epoch improves raw/tuned validation F1 to 0.617/0.644. The original configuration planned two epochs; `stop.json` records the requested stop after epoch one, with optimizer and RNG saved for resumption. FinBERT, BGE and blend predictions were reloaded and checked against validation, including the longest validation document. Final test inference timing runs with the GPU otherwise idle.

```powershell
# Fresh reproduction of the completed one-epoch Laya trial:
python decision_models/train.py --model laya --adaptation lora --lora-rank 8 --epochs 1 --learning-rate .00003 --encoder-learning-rate .0001 --class-weight-power 1 --output decision_models/training_runs/laya-lora
# Optional future second epoch; not part of these frozen selections:
python decision_models/train.py --model laya --adaptation lora --lora-rank 8 --epochs 2 --learning-rate .00003 --encoder-learning-rate .0001 --class-weight-power 1 --output decision_models/training_runs/laya-lora --resume
```

After observing the final test results, further tuning needs a fresh holdout for a clean research comparison. Preserve this run folder before resuming so its frozen checkpoint and decision fingerprints remain auditable.

## Conclusions and rerunning the selected models

Final test results are in [benchmarks.md](benchmarks.md). FinBERT LoRA achieves 0.708 macro F1, character TF-IDF 0.693, the validation-selected blend 0.695, and one-epoch Laya LoRA 0.663. Character TF-IDF is the strongest simple alternative; FinBERT is the most promising encoder adaptation in this comparison. The blend's validation gain did not reproduce clearly on test. No post-test hyperparameter changes were made.

```powershell
# Reproduce FinBERT LoRA from scratch (tuned selection chooses epoch 3):
python decision_models/train.py --model finbert --adaptation lora --lora-rank 8 --epochs 3 --learning-rate .001 --encoder-learning-rate .0001 --class-weight-power 1 --output decision_models/training_runs/finbert-reproduction
# Freeze selection before test inference:
python decision_models/benchmark.py --saved decision_models/training_runs/finbert-reproduction --device cuda --splits decision_models/data/splits
```

Linear searches used C values `.001 .01 .1 1 10 30 100`, with and without balanced class weights. The selected word model uses C=10/unweighted, character C=30/balanced, and frozen FinBERT/BGE C=.001/unweighted. Use `tune.py --model tfidf|char|finbert|bge --regularization .001 .01 .1 1 10 30 100 --output <new-run>` to repeat the search. Exact selected run folders and fingerprints remain locally in `training_runs/final_selection.json`; per-row test results remain in `benchmark_results/`. Checkpoints and failed trials are ignored by Git.

Inference uses one `--saved` interface for linear models, neural heads/adapters and blends, including historical raw checkpoints without decision offsets. Untested full-encoder training and the duplicate head-specific runner were removed. FinBERT reuses its financial-text encoder with a new document-triage classifier, rather than its original sentiment labels.
