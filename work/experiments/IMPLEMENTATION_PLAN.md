# Experiment Harness — Implementation Plan

Goal: squeeze every real point of ROC-AUC out of the ML-08 decline-ranking model by trying
**every applicable method one at a time** — keep what helps, revert what doesn't — then distill
the winning configuration into a single clean submission notebook.

This folder is scaffolding. The deliverable stays `work/notebooks/w05_model.ipynb` (rewritten
from the winning config). Only `.py` / `.md` / `.json*` files here may be committed; every CSV
is gitignored and CI enforces it.

The full method catalogs live in [catalog/](catalog/) — eight files, 188 methods, each with
exact sklearn implementation, leakage analysis, and its own accept test. This document is the
master protocol: what runs, in what order, and how keep/revert gets decided. A future session
should read this file first, then load only the catalog file for the stage it is working on.

---

## 1. Ground truth (verified against the data, 2026-07-15)

| Fact | Value |
|---|---|
| Data | `data/raw/content_refresh_anonymized.csv`, 30,000 rows × 44 cols, 32 clients |
| Population | `impressions_90d >= 100` → 22,006 rows (7,994 sub-threshold rows still have labels) |
| Label | `is_declining = (trend_direction == "down")`, base rate **0.598** (positives are the majority) |
| Metrics | ROC-AUC (primary), Precision@50 (secondary, report-only + veto) |
| Split | client-grouped, always |
| Current best | LogReg pipeline, grouped held-out AUC **0.601**, P@50 **0.84** (w05_model.ipynb) |
| Honest ceiling | ~0.60 AUC; leaky features are banned, chasing AUC with them is disqualifying |
| Environment | Windows, Python 3.14, pandas/numpy/scikit-learn≥1.4 only, `n_jobs=1` mandatory |

`trend_direction` values: down 16,262 / stable 5,962 / up 4,388 / new 2,236 / flat 1,152.
Note: 5 categories, not 3 — the 3-class training-target method must handle all non-down
categories (see catalog/augmentation-sampling.md, adjust `y3` accordingly).

## 2. Column contract

Every config is validated against this table at runtime (`FORBIDDEN` frozenset + assertion in
`build_matrix`). No patch can smuggle a leaky column in.

**Meta (never features):** `content_id` (row id), `client_id` (grouping/weighting only).

**Allowed — static page properties (17):** `search_volume`, `competition`,
`competition_level`, `cpc`, `content_type`, `main_intent`, `word_count`, `char_count`,
`word_count_tier`, `char_count_tier`, `provider_used`, `model_used`, `content_age_days`,
`age_tier`, `age_tier_order`, `days_since_last_update`, `freshness_tier`.

**Allowed — prior window (3):** `impressions_prev_30d`, `clicks_prev_30d`,
`sessions_prev_30d`. Anything derived **only** from allowed columns is allowed (ratios, logs,
ranks, interactions).

**Forbidden as model inputs (20)** — the 90d window overlaps the label window:
`impressions_90d`, `clicks_90d`, `pageviews_90d`, `sessions_90d`, `users_90d`,
`engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`, `days_with_impressions`,
`days_with_sessions`, `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `ctr`,
`avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `impression_tier`,
`position_tier`.

**Label tier (2):** `trend_direction`, `trend_pct`.

Three narrow exceptions, all one-directional:

1. `impressions_90d` may define the **population** (filter / eval mask / train-row weights),
   never a feature. The eval mask is always `impressions_90d >= 100`, fixed.
2. `trend_direction` / `trend_pct` may parameterize **train-fold-only** targets and sample
   weights (3-class target, regression surrogate, confidence weighting) — same information
   tier as the label itself, which the model already sees at fit time. Never a feature, never
   touches eval rows except through the fixed binary label.
3. `client_id` may parameterize train-fold sample weights and label-free per-client derived
   features (ranks, deviations, size) — legitimate because scoring is a per-site batch at
   inference. Never one-hot it, never target-encode it.

## 3. Harness architecture

Detail: [catalog/harness-design.md](catalog/harness-design.md). Summary of what gets built:

```
work/experiments/
  IMPLEMENTATION_PLAN.md      this file
  catalog/                    8 method catalogs (committable)
  run_experiments.py          the whole harness, one file
  ledger.jsonl                append-only trial record (committable)
  lockbox.json                the 6 held-out client ids
  .gitignore                  *.csv, *.pkl, *.npy
```

- **Flat config dict.** One `BASELINE` dict replicating w05 exactly; a candidate is
  `{**best, **patch}`. Keys: `population` (visible / widened variants), `target` (binary /
  3class / regress_trend), `impute`, `features`, `derived`, `missing_indicators`, `cats`,
  `encoder`, `scaler`, `selection`, `sample_weight`, `model`, `model_params`, `calibrate`,
  `ensemble`. `model_params` is replaced wholesale, never deep-merged. Unknown keys raise.
- **`evaluate(cfg)` is pure.** Builds the matrix (row-local derived columns only), then for
  each cached fold fits a full sklearn Pipeline (imputer/encoder/scaler/selector/model all
  inside — nothing is ever fit outside a training fold) and returns per-fold AUC and P@50
  lists. Two calls with the same config return bit-identical results.
- **Fold cache keyed by population.** Folds are materialized once per population variant, so
  every candidate under a given population sees identical folds and all comparisons are
  fold-paired. A population toggle re-evaluates the current best on the new fold set so the
  comparison stays paired.
- **JSONL ledger.** One line per trial: stage, patch name, config diff, config hash, per-fold
  AUCs and P@50s, mean/std, delta, win count, Wilcoxon p (logged, not gating), pooled AUC
  (logged, not gating), decision, one-sentence reason, kept-chain ids. Flush after every
  write. First line is a header with lockbox ids, seeds, and the data file's md5.
- **Resume by replay.** On startup the ledger is loaded into a hash→result dict; already-run
  configs cost zero compute. Killing and rerunning the script continues where it stopped.
- **Crash-safe trials.** A failing candidate logs an error line and the loop continues.
  A candidate family whose first member exceeds 120s gets its remaining grid skipped
  (`skipped_slow` — time-based, never score-based).
- **Determinism.** Every estimator gets `random_state=42, n_jobs=1` explicitly; numpy only
  via seeded `default_rng`; fold seeds are literals. Full rerun changes zero ledger lines.

## 4. Decision protocol

Detail: [catalog/validation-protocol.md](catalog/validation-protocol.md). This section is the
reconciled, authoritative version (the catalog file contains two agents' variants; where they
differ, this section wins).

### 4.1 Evaluation unit — 15 paired folds

Fifteen folds over the **26 dev clients** (lockbox excluded), 3 seeded partitions of 5,
materialized once and cached. Every candidate and the current best are scored on the identical
15 folds; deltas are fold-paired.

**Implementation note (deviation from the original design):** the plan called for
`StratifiedGroupKFold(5, shuffle=True, random_state=s)`, but on this data it turned out to be
seed-invariant — with 26 size-skewed clients the greedy stratified assignment is deterministic,
so all three seeds produced identical folds and the multi-seed set collapsed to five distinct
folds. The harness (`client_fold_map` in `run_experiments.py`) replaces it with a seeded,
size-balanced whole-client assignment: shuffle clients by seed, assign each to the currently
lightest fold. This gives genuine per-seed variation; base-rate stratification is averaged out
across the three seeds rather than enforced per fold. Verified: baseline reproduces at dev-CV
AUC 0.6016, matching the Week-5 notebook's 0.601.

Before adopting this scheme, run the cheap **CV bakeoff** on the frozen baseline only:
StratifiedGroupKFold vs GroupKFold vs GroupShuffleSplit×10 vs LeaveOneGroupOut, 3 seeds each;
pick lowest seed-to-seed std of the mean. Also run GSS 70/30 once to anchor against the
notebook's 0.601. Pre-registered expectation: StratifiedGroupKFold wins.

### 4.2 Accept rule (KEEP iff all three)

1. mean paired ΔAUC over 15 folds ≥ **+0.002**
2. wins in ≥ **9/15** folds
3. P@50 veto passes: NOT (mean P@50 drops > 0.03 AND P@50 delta negative in ≥ 60% of folds)

Ties and sub-threshold gains revert — simpler config wins by default. The threshold is fixed
in advance and never tuned mid-run. P@50 is also the tiebreaker for A/B choices inside a
single toggle, never a reason to accept on its own.

### 4.3 Null-probe calibration (before the loop)

Generate 10 junk candidates (seeded `rng.normal` columns) and 5 permuted-real-feature
candidates; the accept rule must reject ≥ 14/15. If not, raise the threshold to +0.003 or the
win rate to 65%. This calibrates the gate against this dataset's actual fold noise.

### 4.4 Lockbox — 6 clients, one shot

Before anything runs: pick 6 of 32 clients with `default_rng(20260715)`, re-drawing until the
pooled lockbox has ≥ 3,000 visible rows and base rate in [0.55, 0.65]. Persist to
`lockbox.json`. The greedy loop, bakeoffs, and calibration never touch these clients.
Exactly **two** lockbox evaluations, ever: frozen original baseline and final kept-chain, both
trained on all 26 dev clients. If final lockbox AUC < dev-CV mean − 2·(std of fold means),
the report says "greedy chain overfit CV" — the chain is never re-tuned against the lockbox.

### 4.5 Fresh-seed re-verification + prune (after the loop, before the lockbox)

Re-score baseline and final chain on fresh folds (seeds 10, 11, 12). Then leave-one-step-out:
drop each kept step; if removal costs nothing (fresh-fold mean ΔAUC > −0.001), prune the step.
One pass only — no re-running the greedy loop on the fresh folds.

### 4.6 Permutation negative control

Twice (after Stage 2 and after the loop): permute labels **within client**, run the kept chain
through the standard evaluation. AUC > 0.55 ⇒ halt and audit the last kept steps. Expected
~0.50 ± 0.02.

### 4.7 Multi-seed final confirmation

Final config and baseline scored with model `random_state ∈ (42, 43, 44)` on the fresh folds;
report mean ± std. If the seed spread exceeds the claimed gain, the notebook says so.

### 4.8 Budget and stopping

Max **80** candidate evaluations in the greedy phase (~40 min worst case at n_jobs=1); overflow
is logged `skipped_budget`. The retry pass caps at the 10 highest-|Δ| rejected steps and stops
early after 8 consecutive rejections. No third pass, ever. Beam search width 2 gets one trial
on Stage 2 only; adopt only if it beats beam-1 by ≥ +0.002 (expected: it won't).

## 5. Stage order and method index

Fixed order — data-side decisions first, model in the middle, ensembles last, one bounded
re-loop. Within a stage, candidates run cheapest-first; an accepted patch immediately becomes
the new base. Priorities below come from the catalogs (high → run first).

### Stage 0 — Infrastructure (no toggles)

Baseline reproduction (`evaluate(BASELINE)` must land within 0.03 of 0.601), CV bakeoff,
fold cache, lockbox selection, null-probe calibration, ledger header.

### Stage 1 — Preprocessing ([catalog/preprocessing.md](catalog/preprocessing.md), 30 methods)

High: median imputation vs fillna(0) · missing-as-category for all categoricals ·
`has_ai_refresh` binary (provider_used 71% null) · quantile-normal scaler · extended log1p.
Medium: robust scaler · yeo-johnson · quantile-uniform · winsorize grid · ordinal tier
encoding · null-family collapse (search_volume/competition/cpc null together = one flag) ·
universal missing indicators · correlation redundancy pruning (word_count vs char_count) ·
IsolationForest outlier flag · KBins · zero-vs-median for prev counts.
Low: mean impute · per-category median · KNN/Iterative imputers · minmax/maxabs · no-scaling ·
OHE drop variants · frequency encoding · OOF target encoding · rare-category grouping ·
duplicate drop · near-zero variance · outlier removal (train-only).

### Stage 2 — Feature engineering ([catalog/feature-engineering.md](catalog/feature-engineering.md), 28 methods)

High: prev-window ratios (prev CTR, sessions/impression, clicks/session) · zero-prev-traffic
flags · staleness ratio + update recency · keyword opportunity composites (sv×cpc,
competition×cpc, staleness×volume) · log1p all skewed · per-client rank features ·
client aggregate features (n_pages, traffic share, deviations) · spline basis expansion ·
missing-indicator expansion · median-impute variant · one-hot provider/model_used.
Medium: content density ratios · hand-picked pairwise products · full PolynomialFeatures
(then Stage 3 prunes) · KBins · one-hot tiers · tier ordinal · categorical crosses ·
group-wise centering · KMeans distances · Nystroem RBF · quantile transform.
Low: yeo-johnson/sqrt alt · squared-only · frequency encoding · OOF target encoding ·
PCA append · winsorize.

Run the first permutation negative control at the end of this stage.

### Stage 3 — Feature selection ([catalog/feature-selection.md](catalog/feature-selection.md), 19 methods)

Only earns its keep if Stage 2 grew the matrix (with ~30 features L2 already copes; with 400+
poly features it's load-bearing). High: greedy single-feature ablation · poly + univariate
pruning · permutation-importance pruning · correlation pruning. Medium: SFS-forward (grouped
inner CV!) · stability selection (client-level subsample) · L1 SelectFromModel · f_classif
percentile · manual shortlist floor. Low: mutual info · tree importances · RFE/RFECV ·
SBS · variance threshold · PCA replace/augment · TruncatedSVD · FeatureAgglomeration.

### Stage 4 — Population, sampling, training target ([catalog/augmentation-sampling.md](catalog/augmentation-sampling.md), 24 methods)

High: sample weights ∝ prev-window traffic · confidence weighting by |trend_pct| (train folds
only) · per-client rebalancing weights · train-population widening to all 30k rows (eval mask
unchanged!) · widening with down-weighted sub-threshold rows · 3-class training target.
Medium: near-threshold-band widening · regression surrogate (Ridge/HGBRegressor on −trend_pct).
Low (enumerate-and-reject, run cheap): drop class_weight='balanced' · class-weight grid ·
under/oversampling · SMOTE-like · Tomek links · jitter · mixup · bootstrap bagging ·
EasyEnsemble · pseudo-labeling (skip) · recency weights · imputed-count down-weighting · TTA.

Consistency rule: per-client derived features (ranks, aggregates) are recomputed on whatever
population the config trains on; a widening patch invalidates the feature-matrix cache.

### Stage 5 — Models ([catalog/models.md](catalog/models.md), 27 methods)

High: LogReg C-tuning · **HistGradientBoostingClassifier** (native NaN + categorical support —
top candidate; when active, drop imputation/scaling/OHE branches and feed raw + `categorical_features`)
· RandomForest grid. Medium: elasticnet-saga LogReg · LDA (+shrinkage) · ExtraTrees ·
multi-seed averaging. Low: Ridge · SGD · QDA · NB variants · kNN · SVC-rbf (subsample or
Nystroem) · LinearSVC · NuSVC · DecisionTree · GradientBoosting (superseded by Hist) ·
AdaBoost · Bagging(LogReg) · MLP · GaussianProcess (reject on cost).

Critic additions (not in the catalog, add as candidates): HistGB `monotonic_cst` (e.g. force
risk monotone in staleness) · HistGB `interaction_cst` · `RandomTreesEmbedding` + LogReg.
Note: sklearn ≥1.3 has `preprocessing.TargetEncoder` — its internal CV is not group-aware,
same caveat as the manual OOF recipe.

### Stage 6 — Re-loop (conditional)

Only if Stage 5 changed the model family: retry the ≤10 highest-|Δ| rejected steps whose
mechanism is model-dependent (tags: interaction-terms, binning, monotone-transform,
rare-category, feature-add; scaling toggles are linear-only and are not retried). Also re-run
TTA and jitter if the model is now tree-based.

### Stage 7 — Ensembling (from [catalog/models.md](catalog/models.md))

High: soft voting over top models · rank-average blending of top-k. Medium: fold-tuned weighted
blending (leave-one-fold-out for the weights) · manual group-aware OOF stacking (never stock
`StackingClassifier` — its inner CV ignores groups). Low: CalibratedClassifierCV (monotone
calibration can't move a single model's AUC; only relevant inside blends).

### Stage 8 — Verification gauntlet

Fresh-seed re-verification → leave-one-step-out prune → second permutation control →
multi-seed confirmation → the two lockbox shots. Order is fixed; each step's rule is in §4.

### Stage 9 — Final evaluation suite ([catalog/evaluation-analysis.md](catalog/evaluation-analysis.md), 19 methods)

For the notebook: P@K for K ∈ {25, 50, 100, 250} · PR-AUC + ROC curve · lift/gain ·
per-client AUC distribution (worst-client callout) · bootstrap CIs (cluster-by-client and
row-level) · calibration (report-only) · permutation importance on held-out folds ·
coefficient/PDP interpretation · error slicing by content_type/intent/age/freshness/null
pattern · confidently-wrong top-50 inspection · comparison table vs Week-4 CTR-fix and
staleness floor on identical folds · seed-stability line · top-K composition stability ·
below-filter generalization check (curiosity, report-only) · honest-language checklist.

### Stage 10 — Distillation

Rewrite `work/notebooks/w05_model.ipynb`: winning config hardcoded inline, no imports from
this folder. Rubric sections: problem & label → population & leakage policy (the §2 table) →
split design + lockbox → features & preprocessing → method choice + comparison table (with
2-3 sentences on rejected alternatives, citing ledger deltas) → lockbox result → error
analysis → limitations (the honest ~0.60 ceiling). Code cells: no `#` comments, no defensive
guards. Execute with nbclient, exit 0, printed metrics must match the ledger's confirm/lockbox
lines. Prose through the humanizer pass; zero em-dashes; commit without co-author trailer.

## 6. Data audits (run once in Stage 0, log to ledger)

- `days_since_last_update` timing: confirm it is measured at snapshot time (start of label
  window), not inside it — check against the w03 data contract. If ambiguous, note it in the
  notebook's leakage policy section rather than silently assuming.
- `word_count == 0` vs null: 7,699 nulls — verify zeros don't also occur (two different
  "no content data" regimes would need two indicators).
- Confirm `content_id` uniqueness (verified once already: 30,000 unique).
- Client size distribution: motivates per-client rebalancing; log the min/median/max.
- Per-client base rate spread: motivates StratifiedGroupKFold; log it.

## 7. Session workflow (per implementation session)

1. Read this file; load the catalog file(s) for the stage at hand.
2. Fresh venv if jupyter/nbconvert misbehaves; `nbclient` for notebook execution.
3. Run `python work/experiments/run_experiments.py` — it resumes from the ledger.
4. After any notebook rewrite: nbclient run, grep for `#` comments in code cells, em-dashes
   and AI-tells in prose, and (habit, not because tokens are used here) token patterns.
5. Commit only `.py`/`.md`/`.json*` from this folder; never a CSV; no co-author trailer;
   re-run CI on transient "runner not acquired" failures.

## 8. What done looks like

- Ledger shows every method above either evaluated (keep/revert + numbers) or explicitly
  `skipped_budget`/`skipped_slow`/`skip` with a reason.
- Final config beats the frozen baseline on fresh folds and holds up on the lockbox, or the
  notebook honestly reports that the ceiling won and ships the simplest config within 0.002
  of the top.
- One self-contained notebook, executed top-to-bottom, CI green, single contributor.
