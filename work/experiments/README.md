# Experiment harness

This folder holds the search that produced the ML-08 model. The deliverable is the notebook at
[`../notebooks/w05_model.ipynb`](../notebooks/w05_model.ipynb), which stands alone; everything
here is the supporting evidence for how its one modeling decision was reached.

`run_experiments.py` is a greedy harness. Starting from a Logistic Regression baseline that
reproduces the Week-5 result, it tries roughly fifty methods one at a time across preprocessing,
feature engineering, sampling, and twelve model families, scoring each on fifteen paired
client-grouped folds and keeping a change only when it clears an accept rule (mean ΔAUC ≥ +0.002,
wins in at least 9 of 15 folds, and no Precision@50 regression). Six clients are held out as a
lockbox before the search starts and read exactly twice at the end. Run `python run_experiments.py
setup` for the baseline reproduction, lockbox selection, and null-probe calibration; `full` for
the greedy loop; `verify` for the fresh-seed re-check, leave-one-step-out prune, permutation
control, and lockbox shots. Every trial is appended to `ledger.jsonl`, which is the full record of
what was tried and kept or reverted.

The headline result: at the leakage-safe ceiling (~0.60 CV AUC) almost nothing moved the needle
robustly. The one change that survived fresh folds and the lockbox was widening the training
population to include sub-threshold pages, which lifts held-out Precision@50 from 0.76 to 0.84
while holding AUC. No alternative model family beat Logistic Regression as a ranker. The reasoning
and method catalogs are in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and
[`catalog/`](catalog/). CSVs and prediction dumps are gitignored; only the code, plan, ledger, and
lockbox record are tracked.
