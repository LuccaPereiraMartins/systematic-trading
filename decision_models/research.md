# Financial-document triage research programme

Approved scope: source-content-only triage for analyst inboxes, written for AI/ML engineers in finance. Labels remain routine, review_worthy and unclear; unclear routes to review. Macro F1 is primary. Review-worthy -> routine is the critical miss. Trading strategies and historical-context reasoning are outside this study.

## Five dependent PRs

1. **Data**: corporate releases, 6-Ks, Fed, ECB and general news alongside 8-Ks; source census and rights register; provenance; staged Luna labels; temporal/event/near-duplicate separation; blinded human-review pack.
2. **Baselines**: word/character/combined TF-IDF, logistic regression/SVM, frozen encoders; validation-only selection; probability calibration and conservative discard policies; common quality/cost/throughput records.
3. **Decision models**: Laya base/head/LoRA; cross-entropy versus Brier loss; 512/1024/2048/4096 context; fixed existing pooling; sample efficiency; ModernBERT architecture control.
4. **Alternatives**: FinBERT/BGE adaptation, Qwen3-1.7B/4B prompt classification and bounded QLoRA; explicit truncation and matched-coverage comparisons; a cost proposal for the GPT-6.1 Sol reference (not executed).
5. **Release**: source-backed report and PDF, figures, error analysis, paired group bootstrap, reproducibility audit and a small serial experiment runner.

## Frozen defaults

- One seed: 42. No repeated-seed training; evaluation bootstrap does not measure training variability.
- Collect up to 100,000 candidate documents from 2019-01-01 through 2026-09-30. Primary 8-Ks at most 40% of each retained train/validation/test split, including after budget-limited labeling; no pretending that source shortfalls were filled by more 8-Ks. Raw acquisition counts remain separate from benchmark support.
- News coverage includes reusable Wikinews and HM Treasury official announcements, with separate provider counts and metrics. The journalistic cohort has no Q3 2026 coverage; official announcements cannot establish that generalization.
- Corporate releases are a text-query sample of SEC EX-99 exhibits from 8-K filings. The saved search responses verify 244,685 candidates and paginated ranges cover all approved dates. Twelve exhibits from interrupted monthly searches were absent from completed smaller-range retries and remain traceable to their original responses. Discovery does not establish an exhaustive release census.
- Offline re-extraction checks source text, publication dates and response provenance. Three retained Wikinews pilot records had their dates and licence fields repaired from preserved responses; their bodies, identities and training-period membership are unchanged. Original files and verification receipts are retained.
- Train through 2026-03-31, validation 2026-04-01..2026-06-30, test 2026-07-01..2026-09-30. Previously experimented-on bodies and near duplicates cannot enter new validation/test. Related groups crossing boundaries are quarantined. Issuer-disjoint evaluation is outside the core scope.
- Up to 1,200 validation and 1,800 test rows; validation partitions 70% selection/30% calibration and policy fitting. No test inference until all core selections are frozen.
- Nested training subsets: 250, 1000, 4000, 16000 and all available. Start each neural run from its original pretrained checkpoint.
- GPU jobs run serially on the local RTX 3060 12 GB. Preserve checkpoints, source fingerprints and failed configurations. Local API cost is not the same as zero compute cost.
- Only Luna Flex labeling is currently authorised for paid execution: a shared **$3 maximum**, preserving at least $4 of the stated $7 credits. Keep future relabeling possible. Cloud and paid reference benchmarks need concrete quotes before execution.
- Supervised head/LoRA objectives are core. RL/RLCD is a possible follow-on after these five PRs.
- Human review: 12 rubric examples, 72 fresh test documents, 12 overlapping second reviews; three hours total. Prepare now, review later. Unresolved disagreements stay unresolved. Independent quality claims wait for review.

The first overnight/daytime phase covers collection pilots, staged labeling, splits, baselines and initial adaptation screening. The complete experiment matrix is a multi-day programme. Publication claims must distinguish teacher agreement, non-expert human review, missing paid benchmarks and training-seed limitations.
