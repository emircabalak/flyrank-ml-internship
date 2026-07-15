### single_script_layout (priority: high)

**How:** One file: work/experiments/run_experiments.py containing load_data(), build_matrix(cfg), make_pipeline_from(cfg), evaluate(cfg), STAGES, accept(), and the greedy main loop. Beside it: work/experiments/ledger.json (ndjson, append-only), work/experiments/README.md (2-paragraph pointer for reviewers), work/experiments/.gitignore. No package, no per-experiment scripts, no cache directory on disk — dataset is 22k x 44, everything fits in RAM. Candidate patch lists from the other dimensions are plain Python literals inside STAGES in the same file.

**Why:** The harness is throwaway supporting evidence, not a product; one file is trivially resumable, diffable, and committable. Multiple scripts would just duplicate load/eval code.

**Leakage:** none

**Accept:** Structural, always kept. Sanity check: running the script evaluates BASELINE and reproduces the known grouped AUC ~0.60 before any stage starts; abort if |auc_mean - 0.601| > 0.03.

### gitignore_and_committables (priority: medium)

**How:** work/experiments/.gitignore with two lines: *.csv and *.pkl. Committable artifacts are exactly run_experiments.py, ledger.json, README.md. Any prediction dump (oof_final.csv) lands in the same folder and is auto-ignored. The final notebook lives in work/notebooks/ per repo convention, not here.

**Why:** Enforces the repo rule (only .md/.json/.py committed from work/experiments/) mechanically instead of by memory.

**Leakage:** none

**Accept:** Structural, always kept. Check: git status shows no untracked CSVs after a full run.

### flat_config_dict (priority: high)

**How:** BASELINE = {"min_impressions": 100, "impute": "zero", "features": [10 raw numerics], "log1p_prev": True, "derived": [], "missing_indicators": True, "cats": ["content_type","main_intent","competition_level"], "encoder": "onehot", "scaler": "standard", "model": "logreg", "model_params": {"class_weight":"balanced","max_iter":2000,"C":1.0,"random_state":42}, "sampler": None, "calibrate": None, "seed": 42}. A candidate patch is a plain dict; application is cand = {**best, **patch} — model_params is replaced wholesale, never deep-merged, so a params patch must carry the full dict. Every key maps to one branch in build_matrix/make_pipeline_from; unknown keys raise KeyError (fail loud in the harness script).

**Why:** Flat dict + shallow merge keeps every method independently toggleable with zero framework; deep-merge and dataclasses are complexity with no payoff at this scale.

**Leakage:** Config lists ALLOWED columns only; forbidden columns are hardcoded in a FORBIDDEN frozenset and build_matrix asserts features/cats are disjoint from it, so no future patch can smuggle a leaky column in.

**Accept:** Structural, always kept. Check: evaluate(BASELINE) runs; assert set(cfg) == set(BASELINE) on every candidate rejects typo'd patches.

### config_hash_dedup (priority: medium)

**How:** h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]. Computed at the top of evaluate(); if h is already in the RESULTS dict loaded from the ledger, return the stored result without recomputing. Hash also written to every ledger line as "hash", with "base_hash" of the config it was compared against.

**Why:** The greedy loop and the retry pass revisit identical configs (e.g. a patch that equals the current best); dedup makes reruns and resumes free.

**Leakage:** none

**Accept:** Structural, always kept. Check: second invocation of the script recomputes zero trials (all hashes hit).

### fixed_fold_cache (priority: high)

**How:** After the population filter (impressions_90d >= 100) and after removing lockbox clients: cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); FOLD_CACHE[min_impressions] = list(cv.split(np.zeros(len(y)), y, groups=client_id)). Computed once per population value at first request, held in a module-level dict (index arrays over the filtered frame's positional index). Every evaluate() call iterates the exact same (train_idx, test_idx) pairs, so all candidate comparisons are paired on identical folds. ~26 non-lockbox clients across 5 folds gives ~5 clients per test fold.

**Why:** Identical folds are what make per-fold paired acceptance valid; StratifiedGroupKFold balances the 0.598 base rate across folds better than GroupKFold with only 26 groups.

**Leakage:** Stratification uses y to assign whole clients to folds, which is standard and does not leak row-level labels across the group boundary. Folds are keyed by the population filter so a population-changing candidate can never reuse mismatched indices.

**Accept:** Structural, always kept. Check: two consecutive evaluate(BASELINE) calls return bit-identical auc_folds lists.

### lockbox_client_holdout (priority: high)

**How:** clients = sorted(df.client_id.unique()); LOCKBOX = set(np.random.default_rng(2026).choice(clients, 6, replace=False)). Rows from lockbox clients are dropped before fold construction and are never touched by the greedy loop. At the very end, fit the winning pipeline on all non-lockbox filtered rows and score the lockbox filtered rows exactly once: ROC-AUC plus P@50; write one ledger line with stage="lockbox". The rng seed 2026 is a literal at the top of the script and the chosen client ids are echoed into the ledger header line so the set is auditable and frozen.

**Why:** The greedy loop runs ~50-80 trials against the same 5 folds and will adaptively overfit them by a few thousandths of AUC; 6 never-seen clients give one honest generalization number for the notebook.

**Leakage:** none — this is the anti-leakage/anti-overfitting mechanism. The one rule: evaluate lockbox exactly once; a second look after tweaking would un-lock the box.

**Accept:** Structural, always kept. Interpretation rule written in advance: if lockbox AUC is within ~0.03 of CV mean, the config generalizes; if it collapses toward 0.5, report the CV number with that caveat rather than iterating on the lockbox.

### evaluate_contract (priority: high)

**How:** def evaluate(cfg) -> {"hash", "auc_folds": [5 floats], "auc_mean", "auc_std", "p50_folds": [5 floats], "p50_mean", "seconds"}. Body: X, y, groups = build_matrix(cfg) (raw + row-wise derived columns only); for tr, te in FOLD_CACHE[cfg["min_impressions"]]: pipe = make_pipeline_from(cfg) — a ColumnTransformer(SimpleImputer/encoder per cfg) + scaler + model Pipeline; pipe.fit(X.iloc[tr], y.iloc[tr]); p = pipe.predict_proba(X.iloc[te])[:,1]; roc_auc_score and P@50 per fold. Pure function of cfg: all fitting (imputation stats, encoder categories, scaler means, any target encoding) happens inside the Pipeline on train indices only. All estimators get n_jobs=1 and random_state=42 explicitly.

**Why:** A single pure entry point is the whole harness contract: driver, retry pass, multi-seed confirmation, and lockbox all reuse it, and purity is what makes ledger-replay resumability correct.

**Leakage:** The critical rule lives here: NO preprocessing fit outside the fold loop. fillna/scaling/encoding as pipeline steps, never applied to the full frame. Row-wise derived features (e.g. prev CTR) are computed pre-fold, which is safe because they use no cross-row statistics.

**Accept:** Structural, always kept. Check: evaluate(BASELINE) matches the w05 notebook's held-out AUC within noise (~0.60) and takes <10s.

### in_memory_feature_matrix_cache (priority: low)

**How:** FEATMAT = {} module dict keyed by md5 of json.dumps of the matrix-relevant sub-config (min_impressions, features, derived, log1p_prev, missing_indicators, cats). build_matrix checks it first; stores (X, y, groups) as pandas objects. Never cache fitted transformers — only the raw/derived column frame.

**Why:** Honest note: at 22k rows the matrix build is well under a second, so this saves almost nothing; included because the contract asks for it and it is 5 lines. Expect no measurable effect.

**Leakage:** Safe only because cached content is row-wise raw/derived columns; anything fold-dependent (imputation, encoding, scaling) stays in the pipeline. Caching a fitted transform here would be train/test contamination — the key-set definition prevents it.

**Accept:** Structural, always kept. Check: cached vs uncached evaluate(BASELINE) produce identical auc_folds.

### precision_at_50_metric (priority: high)

**How:** def p_at_50(y_true, proba): return float(np.asarray(y_true)[np.argsort(-proba)[:50]].mean()). Computed per fold inside evaluate (test folds are ~3-5k rows, so top-50 is well defined) and once on the lockbox. Reported in every ledger line; never used for acceptance.

**Why:** It is the task's secondary metric and the notebook must report it; with base rate 0.598 and baseline P@50 0.84 it mostly saturates, so gating on it would just add noise to keep/revert decisions.

**Leakage:** none

**Accept:** Structural, always kept — report-only by design; acceptance is AUC-only (see paired_fold_acceptance_rule).

### ndjson_ledger (priority: high)

**How:** work/experiments/ledger.json, one JSON object per line, opened in append mode with flush after every write: {"ts": iso timestamp, "stage": str, "patch": patch-name, "hash": config_hash, "base_hash": current-best hash, "auc_mean", "auc_std", "auc_folds", "p50_mean", "delta": auc_mean - best_auc_mean, "wins": paired fold wins out of 5, "accepted": bool, "seconds": float, "config": full cfg dict}. First line is a header record with lockbox client ids, fold seed, and data file md5. Special stages: "baseline", "retry", "confirm", "lockbox".

**Why:** The ledger IS the experiment record: it drives resumability and dedup, and it is the committable (.json) evidence trail the notebook's comparison table is generated from.

**Leakage:** none

**Accept:** Structural, always kept. Check: json.loads succeeds on every line after a full run; exactly one accepted=True line per kept step.

### resume_by_ledger_replay (priority: medium)

**How:** At startup: RESULTS = {rec["hash"]: rec for rec in map(json.loads, open(ledger))} (skip header/error lines). evaluate() returns RESULTS[h] on hit. Because STAGES order and patch lists are deterministic and acceptance is a pure function of stored fold scores, rerunning the script after a crash replays all completed trials from the ledger at zero compute and continues at the first missing hash. No pickled loop state, no checkpoint files.

**Why:** Crash/interrupt recovery for free; also means adding a new candidate to a stage and rerunning only computes the new one.

**Leakage:** none

**Accept:** Structural, always kept. Check: kill the script mid-stage, rerun, final best hash identical to an uninterrupted run.

### greedy_stage_driver (priority: high)

**How:** STAGES = [("impute", [...]), ("derived_features", [...]), ("encoding", [...]), ("scaling", [...]), ("model", [...]), ("model_params", [...]), ("imbalance", [...]), ("calibration", [...])] — each entry a list of (name, patch_dict) supplied by the other dimensions. Loop: best, best_res = BASELINE, evaluate(BASELINE); for stage, patches in STAGES: for name, patch in patches: cand = {**best, **patch}; res = evaluate(cand); log; if accept(best_res, res): best, best_res = cand, res. Sequential within a stage: an accepted patch becomes the new base immediately, so later candidates in the same stage compete against it. Stage order is fixed and coarse-to-fine: data first, model in the middle, params and calibration last.

**Why:** Greedy one-at-a-time is exactly the harness premise; ~60-80 candidates at <=30s each keeps the whole search under ~40 minutes at n_jobs=1. Honest note: greedy misses interactions between stages — the retry pass below recovers the main one (feature x model).

**Leakage:** none at the mechanism level; adaptive CV overfitting across many trials is the residual risk, bounded by the lockbox.

**Accept:** This IS the keep/revert machinery; per-candidate decisions delegate to paired_fold_acceptance_rule.

### paired_fold_acceptance_rule (priority: high)

**How:** def accept(base_res, cand_res): delta = cand_res["auc_mean"] - base_res["auc_mean"]; wins = sum(c > b for c, b in zip(cand_res["auc_folds"], base_res["auc_folds"])); return delta >= 0.002 and wins >= 3. Valid as a paired comparison because folds are identical for every config. Epsilon 0.002 chosen against observed fold std (~0.01-0.03 with 5 client-grouped folds). Ties and sub-epsilon gains revert — simpler config wins by default. No Wilcoxon/t-test: 5 folds is too few for it to mean anything, so a win-count heuristic is the honest version.

**Why:** The single most important guard against ratcheting up noise: with a low ceiling (~0.60) most candidate deltas will be within fold noise, and accepting them would random-walk the config.

**Leakage:** Mild adaptive overfitting to the fixed folds is inherent to any greedy CV search; mitigated by epsilon, win-count, multi-seed confirmation, and the lockbox.

**Accept:** Structural, always kept; the epsilon itself is fixed in advance and never tuned mid-run (tuning the acceptance threshold on results would be selection leakage).

### rejected_retry_after_model_switch (priority: medium)

**How:** Driver accumulates rejected = [(stage, name, patch)] for all pre-model stages. After the "model" and "model_params" stages complete, if best["model"] != BASELINE["model"], run one extra pass: for each rejected patch, cand = {**best, **patch}; evaluate; same accept rule; ledger stage="retry". Exactly one pass, no recursion — a patch rejected twice stays rejected.

**Why:** The main greedy blind spot here: features and transforms useless to LogisticRegression (monotone transforms, raw counts, interaction-shaped signals) can help HistGradientBoostingClassifier and vice versa. One retry pass captures this for ~15-25 extra evaluations, a few minutes.

**Leakage:** none beyond the general adaptive-CV risk already covered.

**Accept:** Retried patches use the identical paired_fold_acceptance_rule against the post-model-switch best.

### crash_safe_trial_wrapper (priority: low)

**How:** In the driver only: try: res = evaluate(cand) except Exception as e: ledger line {"stage": stage, "patch": name, "error": repr(e), "accepted": False}; continue. The style rule (no defensive guards) binds the final notebook cells, not the harness script.

**Why:** One bad candidate (e.g. IterativeImputer non-convergence, an encoder choking on an unseen category in some fold) must not kill a 40-minute run. Cheap insurance.

**Leakage:** none

**Accept:** Structural, always kept. Check: inject a deliberately broken patch, confirm the run completes and logs the error line.

### runtime_budget_logging (priority: low)

**How:** t0 = time.perf_counter() around each evaluate; "seconds" into every ledger line. Soft budget: print a warning when a trial exceeds 60s; a candidate family whose first member exceeds 120s gets its remaining grid members skipped with logged stage="skipped_slow" lines. Total budget sanity: ~80 trials x <=30s ≈ 40 min worst case at n_jobs=1.

**Why:** Windows n_jobs=1 makes slow candidates (IterativeImputer, big HGB grids, CalibratedClassifierCV) the only real schedule risk; the ledger timing data is also what justifies grid sizes in the notebook write-up.

**Leakage:** none

**Accept:** Structural, always kept; the skip rule is time-based, never score-based, so it cannot bias selection toward lucky configs.

### multi_seed_final_confirmation (priority: high)

**How:** After all stages and the retry pass, before the lockbox: for s in range(5): folds_s = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(zeros, y, groups)); score BOTH the winning config and BASELINE on the same folds_s. Yields 25 paired fold AUCs per config; ledger stage="confirm" with per-seed means. The winner is confirmed if its 25-fold mean beats baseline's by >= 0.002; if not, fall back to the simplest config within 0.002 of the top (report this rule in the notebook). ~10 extra pipeline fits x 5 folds, a few minutes.

**Why:** Guards against the winning config being an artifact of the one fixed fold assignment — the cheap version of nested CV, which the compute budget cannot afford.

**Leakage:** none; still never touches lockbox clients.

**Accept:** This step is itself an accept gate for the final config: pass = keep winner, fail = fall back to baseline-plus-confirmed-steps. Decision rule fixed before running.

### distillation_notebook_spec (priority: high)

**How:** New self-contained work/notebooks/w05_model.ipynb (rewritten in place): reads only data/raw/content_refresh_anonymized.csv, hardcodes the winning config as an inline dict (no imports from work/experiments/). Markdown-section skeleton mapped to the ML-08 rubric: (1) Problem & label definition, (2) Population filter & leakage policy — table of all 23 forbidden columns with one-line reasons, (3) Split design — why client-grouped, fold scheme, lockbox description, (4) Features & preprocessing — final feature table with null counts, (5) Method choice & comparison — table Week-4 LogReg baseline vs winner: AUC mean±std, per-fold AUCs, P@50, plus 2-3 sentences on rejected alternatives citing ledger deltas, (6) Lockbox result, (7) Error analysis — per-client AUC bar chart, top-50 predicted-decline false positives broken down by content_type/age_tier, sklearn.calibration.CalibrationDisplay.from_predictions, (8) Limitations — the ~0.60 honest ceiling and why. Code cells: no # comments, no defensive guards; all narration in markdown cells. The harness folder is referenced by path as supporting evidence only.

**Why:** The notebook is the actual deliverable; the harness is scaffolding. Rubric sections enumerated up front so distillation is mechanical transcription from the ledger, not a rewrite.

**Leakage:** The forbidden-columns table doubles as the notebook's own leakage audit; the notebook recomputes final metrics from scratch so no stale ledger numbers are pasted.

**Accept:** Kept when it satisfies nbclient_execution_check and its recomputed AUC/P@50 match the ledger's confirm/lockbox lines to ~1e-6.

### nbclient_execution_check (priority: medium)

**How:** After writing the notebook: run `jupyter execute work/notebooks/w05_model.ipynb` (nbclient under Python 3.14) from the repo root; require exit code 0; then read the executed output cells and diff the printed AUC/P@50 against ledger stage="confirm" and stage="lockbox" values. Only commit after a clean top-to-bottom run.

**Why:** The single check that the deliverable stands alone — catches hidden-state cells, path assumptions, and version drift that notebook editing always introduces.

**Leakage:** none

**Accept:** Binary gate: exit 0 and matching metrics = done; anything else = fix the notebook, never the numbers.

### determinism_conventions (priority: medium)

**How:** Global rules enforced in make_pipeline_from: every estimator that accepts them gets random_state=42 and n_jobs=1 explicitly (LogisticRegression, HistGradientBoostingClassifier, RandomForestClassifier, CalibratedClassifierCV's inner CV via fixed KFold, any sampler). No use of the global numpy RNG anywhere — only seeded np.random.default_rng instances for lockbox selection. Fold seeds are literals (42 for the main folds, 0-4 for confirmation, 2026 for lockbox).

**Why:** Determinism is what makes hash-based dedup, ledger replay, and paired fold comparisons exact rather than approximate; on Windows n_jobs=1 is also the mandated workaround for the joblib bug.

**Leakage:** none

**Accept:** Structural, always kept. Check: full rerun of the script from an intact ledger changes zero ledger lines.

