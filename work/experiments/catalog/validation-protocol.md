### cv_backbone_stratified_group_kfold_k5 (priority: high)

**How:** Primary CV scheme for every greedy toggle: sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed), groups=df['client_id'], y=is_declining. 5 folds => ~6-7 held-out clients (~4400 test rows) per fold, both classes guaranteed present per fold (needed for roc_auc_score), class fraction balanced across folds despite the 0.598 base rate varying by client. Cost: 5 fits per candidate; LogReg on 22k x ~30 features runs in seconds, HistGradientBoostingClassifier under a minute. Code: skf = StratifiedGroupKFold(5, shuffle=True, random_state=s); folds = list(skf.split(X, y, groups)). Bakeoff protocol: run the FROZEN baseline config under each candidate scheme (this one, GroupKFold, GSS-repeats, LOGO) with 3 seeds each; log mean AUC, per-fold std, and seed-to-seed std of the mean; pick the scheme with lowest seed-to-seed std of the mean under a <60s-per-candidate LogReg budget. Pre-registered expectation: this scheme wins.

**Why:** Grouped folds over 32 heterogeneous clients are the dominant noise source at a 0.60 ceiling; stratification cuts fold-to-fold base-rate variance versus plain GroupKFold, which directly shrinks per-fold AUC variance and makes the accept rule less noisy. This is the load-bearing choice of the whole harness.

**Leakage:** None from the scheme itself. Protocol-level guard it enforces: ALL preprocessing (imputers, scaler, one-hot) must live inside a sklearn Pipeline/ColumnTransformer so it is refit per training fold; any fillna/scaling fit on the full frame before splitting leaks distributional info across clients.

**Accept:** Adopted via the scheme bakeoff on the frozen baseline (never on candidates): lowest seed-to-seed std of mean AUC per unit runtime. Not a greedy toggle — decided once before the loop and then frozen.

### cv_arm_plain_group_kfold_k5 (priority: medium)

**How:** Bakeoff arm: sklearn.model_selection.GroupKFold(n_splits=5). Note sklearn's GroupKFold is deterministic (balances group sizes, no shuffle/random_state until sklearn 1.6's shuffle param; with >=1.6 pass shuffle=True, random_state=s to get seed-able variants, otherwise permute client_id->fold assignment manually with np.random.default_rng(s).permutation over unique clients). Same 3-seed bakeoff measurement as the backbone.

**Why:** Cheapest honest comparator to StratifiedGroupKFold. Expected to lose: without stratification some folds land clients with extreme base rates (a fold that is 0.75 positive yields structurally different AUCs), inflating variance. Worth running because if it ties, it removes the (minor) stratification dependency.

**Leakage:** none

**Accept:** Same bakeoff criterion as the backbone: only adopted if its seed-to-seed std of mean baseline AUC beats StratifiedGroupKFold. Expect revert.

### cv_arm_leave_one_group_out (priority: low)

**How:** Bakeoff arm: sklearn.model_selection.LeaveOneGroupOut() over the 32 clients => 32 fits per candidate (~6.4x the k=5 cost). Per-client AUC is undefined when a client is single-class (roc_auc_score raises ValueError) — must either skip those clients (biases the mean) or score pooled over all 32 held-out predictions (one pooled AUC, so no per-fold distribution for the accept rule). Sketch: for tr, te in LeaveOneGroupOut().split(X, y, groups): fit; collect (y[te], proba[te]); then pooled roc_auc_score on the concatenation.

**Why:** Included for completeness; expected to fail the bakeoff. Per-client AUCs on small clients are extremely noisy, single-class clients break per-fold scoring, and pooled-only scoring starves the win-rate accept rule of paired fold deltas. 6.4x cost buys worse decision statistics.

**Leakage:** none

**Accept:** Bakeoff criterion; additionally auto-reject if any client is single-class (check df.groupby('client_id')['is_declining'].nunique() before running). Expect revert.

### cv_arm_group_shuffle_split_repeats (priority: medium)

**How:** Bakeoff arm: sklearn.model_selection.GroupShuffleSplit(n_splits=10, test_size=0.30, random_state=s) => 10 overlapping 70/30 client splits per candidate. Matches the current notebook's split style so deltas versus the existing 0.601 are directly interpretable.

**Why:** Overlapping test sets make the 10 'fold' scores positively correlated, so naive std underestimates true uncertainty (classic Nadeau-Bengio problem) — the accept rule would be overconfident. Keep only as a fallback if k-fold variants prove too coarse (each of only 5 folds hiding a pathological client), and as the bridge metric to the notebook baseline.

**Leakage:** none

**Accept:** Bakeoff criterion. If it loses (expected), still run it ONCE on the frozen baseline to anchor the harness numbers to the notebook's 0.601, then never again.

### multi_seed_fold_set_15_folds (priority: high)

**How:** The harness's actual evaluation unit: union of StratifiedGroupKFold(5, shuffle=True, random_state=s) for s in (0, 1, 2) => 15 (train, test) index pairs, computed ONCE and cached (see fixed_fold_cache). Every candidate is scored on all 15; deltas versus the current config are computed per fold. Cost: 15 LogReg fits per candidate ~= seconds; 15 HistGB fits ~= 1-3 min. If HistGB becomes the base model and the loop feels slow, drop to seeds (0, 1) = 10 folds.

**Why:** 5 paired deltas is too few to separate a +0.002 real gain from grouped-fold noise; 15 paired deltas gives the win-rate and Wilcoxon rules actual power while staying inside the minutes budget. This is the single best lever against greedy noise-chasing.

**Leakage:** none

**Accept:** Not a toggle; part of the fixed protocol. Sanity check before adoption: std of the 15 baseline fold AUCs should be < ~0.05, and the 3 per-seed means should agree within ~0.01 — if not, escalate to 5 seeds.

### fixed_fold_cache_paired_evaluation (priority: high)

**How:** Materialize fold indices once, before the loop, on dev clients only: folds = [(tr, te) for s in (0,1,2) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X_dev, y_dev, groups_dev)]; np.save each index array (or pickle the list) to the scratchpad/artifacts dir. Every candidate AND the current-best config are always scored on these identical indices, with all model random_state fixed (LogisticRegression is deterministic; HistGradientBoostingClassifier(random_state=42); any SelectFromModel/permutation_importance seeded). Deltas are then fold-paired by construction: delta_f = auc_cand_f - auc_curr_f.

**Why:** Pairing removes the between-fold variance component (which dwarfs any 0.002 signal here) from the accept decision — unpaired comparisons across re-drawn splits would need 10x more folds for the same power. Also makes every ledger row exactly reproducible.

**Leakage:** None across candidates. One subtlety: the SAME folds reused for dozens of sequential decisions is what makes greedy overfitting-to-CV possible — that is precisely what the lockbox and fresh-seed re-verification methods exist to catch; do not 'fix' it by re-drawing folds per candidate (that destroys pairing).

**Accept:** Not a toggle; fixed protocol infrastructure.

### accept_rule_combined_mean_and_winrate_RECOMMENDED (priority: high)

**How:** KEEP a candidate iff ALL of: (1) mean paired delta AUC over the 15 folds >= +0.002; (2) win rate: (deltas > 0).sum() >= 9 of 15 (60%); (3) P@50 veto passes (see p50_veto method). Otherwise REVERT. Implementation: deltas = cand_fold_aucs - curr_fold_aucs; keep = deltas.mean() >= 0.002 and (deltas > 0).mean() >= 0.60 and p50_ok. Recommended over the alternatives because: mean-only is fooled by one lucky fold; Wilcoxon at n=15 alpha 0.1 is nearly equivalent to the win-rate condition but adds a scipy-rank fragility for zero deltas; the conjunction is trivially explainable in the ledger ('mean +0.0031, 11/15 folds, p50 flat -> keep').

**Why:** At a 0.60 ceiling almost every candidate is near-null; the harness's real enemy is accumulating 10 fake +0.001s. The conjunction controls false-accepts far better than any single condition while never rejecting a genuine +0.005 (a real gain of that size wins >60% of paired folds essentially always).

**Leakage:** none

**Accept:** This IS the accept rule. Calibrate it once before the loop with null probes: generate 10 candidates that add rng.normal(size=n) junk feature columns (np.random.default_rng(k)) and 5 candidates that permute one real allowed feature within client; the rule must reject >= 14 of 15 probes. If it accepts more, raise threshold to +0.003 or win-rate to 65%.

### accept_rule_arm_mean_delta_only (priority: low)

**How:** KEEP iff mean paired delta AUC >= +0.002, no other condition. One line: deltas.mean() >= 0.002.

**Why:** Bakeoff arm to demonstrate the combined rule earns its extra condition. Expected to fail the null-probe calibration: with fold std ~0.02, a single +0.05 fluke fold can push the mean of 15 past +0.002 on pure noise.

**Leakage:** none

**Accept:** Run through the same 15-null-probe calibration as the recommended rule; adopt only if it matches the combined rule's false-accept rate (expect it will not).

### accept_rule_arm_wilcoxon_paired (priority: medium)

**How:** KEEP iff scipy.stats.wilcoxon(deltas, alternative='greater', zero_method='zsplit').pvalue < 0.10 (scipy is available — it is a hard dependency of scikit-learn, no new package). n=15 paired deltas gives usable power; at n=5 the minimum attainable one-sided p is 0.03125 so alpha 0.1 effectively demands 4-5/5 wins — another reason the 15-fold set matters.

**Why:** Statistically the most principled single rule and robust to one outlier fold, but adds nothing over mean+winrate in practice at n=15, and the p-value is harder to eyeball in the ledger than 'mean +0.003, 11/15'. Kept as an arm; also cheap to LOG for every decision even if not the gate.

**Leakage:** none

**Accept:** Same null-probe calibration bakeoff as the recommended rule. Even if not adopted as the gate, log its p-value in every ledger row for free post-hoc auditing.

### accept_rule_arm_one_se (priority: low)

**How:** KEEP iff deltas.mean() >= deltas.std(ddof=1) / np.sqrt(len(deltas)) (mean exceeds 1 standard error of the paired deltas). Parameter-free variant of the threshold rule; equivalent to a one-sided t-test at roughly alpha 0.16.

**Why:** Attractive because it adapts to the observed noise instead of a hardcoded +0.002, but it accepts vanishingly small gains when deltas happen to be uniform (mean +0.0004, SE 0.0003 -> keep), bloating the kept-chain with steps that don't survive re-verification. Expect it to lose the calibration to the combined rule.

**Leakage:** none

**Accept:** Null-probe calibration bakeoff; additionally require it to also reject 'micro-gain' probes (real feature duplicated with tiny noise) that the combined rule's absolute +0.002 floor filters out.

### p50_veto_and_tiebreaker (priority: high)

**How:** Per fold, compute P@50 on the held-out clients: order = np.argsort(-proba_te); p50 = y_te[order[:50]].mean() (test folds have ~4400 rows, top-50 is well-defined). VETO: even if the AUC gate passes, REVERT any candidate with mean_p50_cand < mean_p50_curr - 0.03 AND (p50_deltas < 0) in >= 60% of folds (both conditions, so one weird fold can't veto a good step; -0.03 = losing 1.5 true positives per 50, material for the 'which pages do we refresh' use case). TIEBREAKER: used ONLY for A/B choices inside one toggle (e.g., choosing between two model candidates whose AUC gate results are identical): pick higher mean P@50. Never accept a step on P@50 alone — P@50 on 50 items is far noisier than AUC on 4400.

**Why:** P@50 is the business metric (0.84 currently); a candidate that improves global ranking but scrambles the extreme top would be a real regression the AUC gate is blind to. The dual condition keeps the veto from firing on top-50 sampling noise (std of P@50 at n=50, p~0.85 is ~0.05 per fold).

**Leakage:** none

**Accept:** This IS the veto rule; fixed protocol. Sanity-check its false-veto rate on the null probes (a junk feature that passes nothing should not be recorded as 'vetoed', but as 'AUC-rejected') so ledger reasons stay clean.

### lockbox_holdout_6_clients (priority: high)

**How:** BEFORE any harness run: choose 6 of 32 clients (~19%) as a lockbox, using only client_id, row counts, and client base rate for a sanity band — nothing else. Selection: rng = np.random.default_rng(20260715); for seed_try in itertools.count(): pick = rng.choice(sorted(unique_clients), 6, replace=False); accept the first pick where pooled lockbox rows >= 3000 and pooled base rate in [0.55, 0.65] (so the final report is not on a freak subsample). Persist the 6 ids to lockbox.json. The greedy loop, the fold cache, the bakeoffs, the null-probe calibration ALL run on the remaining 26 dev clients only. FINAL-REPORT PROTOCOL: exactly two lockbox evaluations, ever: (a) frozen original baseline config, (b) final kept-chain config — each trained on all 26 dev clients, scored pooled on the 6 lockbox clients (AUC + P@50). Report both numbers next to their dev-CV means. If final lockbox AUC < dev-CV mean - 2 * dev-CV std-of-fold-means, state 'greedy chain overfit CV' in the report; the chain is NOT re-tuned against the lockbox under any circumstances — one shot, then the lockbox is burned.

**Why:** The fixed 15-fold set is reused for every sequential decision, so the final CV number is biased upward by selection; the lockbox is the only unbiased estimate of the whole PROCEDURE. With ~0.60 ceiling and 32 clients this bias can plausibly be the entire reported 'gain', so this is non-negotiable.

**Leakage:** The method exists to PREVENT adaptive leakage. Residual risk: peeking at lockbox base rate for the sanity band is mild adaptive contamination — acceptable because it uses only the label marginal at client granularity, never features; do not tighten the band beyond [0.55, 0.65].

**Accept:** Not a toggle; fixed protocol. Its 'decision output' is the final-report comparison and the overfit flag defined above.

### kept_chain_reverification_fresh_seeds (priority: high)

**How:** After the greedy loop finishes (and BEFORE the lockbox shot), re-score exactly two configs — original baseline and final kept-chain — on a FRESH 15-fold set built from unused seeds: StratifiedGroupKFold(5, shuffle=True, random_state=s) for s in (10, 11, 12), same 26 dev clients. Additionally run a leave-one-step-out pass over the kept chain: for each kept step k, score the chain minus k on the fresh folds; any step whose removal changes fresh-fold mean AUC by > -0.001 (i.e., removal costs nothing) gets PRUNED. Cost: (2 + n_kept) x 15 fits, a few minutes.

**Why:** Steps accepted on folds (0,1,2) partly fit those folds' noise; fresh seeds give an honest pre-lockbox estimate and the leave-one-out prune ejects passenger steps that only looked good jointly. Expect the fresh-fold gain to be smaller than the greedy-fold gain — report both deltas so the shrinkage is visible.

**Leakage:** None. Do not iterate: one prune pass only. Re-running the greedy loop against the fresh folds would just start overfitting the new folds.

**Accept:** Prune rule as stated (removal cost > -0.001 => drop step). The surviving chain is what goes to the lockbox.

### stage_ordering_with_single_reloop (priority: high)

**How:** Fixed stage order: (1) preprocessing/imputation toggles -> (2) feature additions/derivations -> (3) feature selection -> (4) model swap (LogReg alternatives, HistGB, calibration) -> (5) ensemble/blend. Within a stage, evaluate candidates in a pre-registered order (cheapest first) and greedily keep/revert one at a time. RELOOP RULE: exactly one extra pass, and only if stage 4 changed the model family — rerun stages 1-3's REJECTED candidates under the new model (see retry_rejected method); accepted steps are never re-litigated except by the leave-one-step-out prune. No third pass ever.

**Why:** Preprocessing gains are model-agnostic so decide them first; features depend on preprocessing; model choice interacts with everything so it goes late; a bounded single re-loop captures the big LogReg-vs-tree interaction without turning greedy search into an open-ended combinatorial crawl that overfits the folds.

**Leakage:** none

**Accept:** Not a toggle; fixed protocol. The reloop's individual re-tried candidates each face the normal accept rule.

### retry_rejected_steps_after_model_switch (priority: high)

**How:** The ledger keeps every rejected step's config diff. If stage 4 replaces LogisticRegression (e.g., with HistGradientBoostingClassifier(random_state=42, early_stopping=False)), re-queue the rejected steps whose mechanism is model-dependent, filtered by a static tag on each candidate: retry tags = {'interaction-terms', 'binning', 'monotone-transform', 'rare-category-grouping', 'feature-add'} (trees find their own splits/interactions, so a ratio feature rejected under LogReg can still win; conversely scaling toggles are tagged 'linear-only' and are NOT retried under trees, and log1p transforms are near-no-ops for trees so deprioritize them). Each retry is a normal single toggle against the new current-best under the standard accept rule. Budget cap: retry at most the 10 highest-|delta| rejected steps.

**Why:** This is where greedy's biggest blind spot lives: a prev_ctr-style ratio or a raw-count feature can be useless through a linear link but valuable to a tree. Cheap to exploit because everything is already implemented and cached.

**Leakage:** None new; each retried step was already leakage-vetted when first proposed.

**Accept:** Standard combined accept rule per retried candidate.

### beam_search_width_2 (priority: low)

**How:** Instead of one current-best config, carry the top-2 configs after each stage; evaluate every remaining candidate against both beams (2x fold-evaluation cost), keep the two best (config, mean AUC) pairs, collapse to the winner before stage 4. Implementation is a list of (config_dict, fold_aucs) sorted by mean.

**Why:** Expected NOT worth it here and included so the harness can reject it cheaply on a single stage: with near-null candidates at a 0.60 ceiling, beam-2 mostly carries a noise-twin of beam-1 and doubles runtime; beam search pays off when steps interact strongly, which the single reloop already covers at lower cost. Honest expectation: revert.

**Leakage:** none

**Accept:** Trial on stage 2 only: run stage 2 with beam-1 and beam-2; adopt beam-2 for later stages only if its stage-end best config beats beam-1's by >= +0.002 mean AUC on the same folds. Expect revert.

### seed_policy_and_final_variance_report (priority: high)

**How:** Fix every stochastic knob globally: split seeds (0,1,2) for greedy, (10,11,12) for re-verification; model random_state=42 everywhere (HistGB, any SelectFromModel estimator, permutation_importance(random_state=42, n_jobs=1)); lockbox selection seed 20260715; null-probe rng seeds 100..114. numpy ops via np.random.default_rng(seed), never global np.random.seed. FINAL REPORT: for the final config, train/eval with model random_state in (42, 43, 44) on the fresh fold set and report mean +/- std over the 3 model seeds x 15 folds — LogReg contributes ~0 variance (deterministic), HistGB a real spread; if std-over-model-seeds > the reported gain, say so explicitly in the report.

**Why:** n_jobs=1 is already mandated; full seed pinning makes every ledger row re-runnable byte-for-byte, and the 3-model-seed spread stops the final writeup claiming +0.004 when HistGB's own seed jitter is +/-0.005.

**Leakage:** none

**Accept:** Not a toggle; fixed protocol.

### permutation_negative_control (priority: high)

**How:** Leakage smoke test run twice (after stage 2 and after the loop ends): y_perm = df.groupby('client_id')['is_declining'].transform(lambda s: pd.Series(np.random.default_rng(0).permutation(s.values), index=s.index)) — permuting WITHIN client preserves client base rates so client-identity shortcuts can't masquerade as signal. Run the current kept-chain through the standard 15-fold evaluation against y_perm. HALT criterion: mean AUC > 0.55 => a kept step is leaking (or the pipeline fits something outside the folds); freeze the loop and audit the last kept steps before continuing. Expected result ~0.50 +/- 0.02. Cost: one candidate-evaluation's worth of compute per run.

**Why:** The forbidden-column list guards known leaks; this guards unknown ones introduced by derived features or preprocessing mistakes (e.g., an imputer accidentally fit pre-split). Two minutes of compute for the strongest possible tripwire.

**Leakage:** This method is itself the leakage detector; permutation within client is required — global permutation would let true client-level base-rate structure inflate the control.

**Accept:** Gate, not a toggle: AUC on permuted labels must stay <= 0.55 or the harness halts for audit.

### results_ledger_jsonl (priority: high)

**How:** Append one JSON line per candidate evaluation to work/harness_ledger.jsonl, written via json.dumps then f.write(line + '\n') immediately after each decision (crash-safe). Schema per line: {"step": int (monotonic, this IS the ordering — no timestamps), "stage": "preprocessing|features|selection|model|ensemble|reverify|control|lockbox", "candidate": "short_id", "config_diff": {json-able dict, e.g. {"add_feature": "prev_ctr", "formula": "clicks_prev_30d/np.maximum(impressions_prev_30d,1)"}}, "fold_aucs": [15 floats], "fold_p50s": [15 floats], "fold_clients": [[client ids per test fold]], "mean_auc": f, "std_auc": f, "delta_auc": f, "win_rate": f, "wilcoxon_p": f, "mean_p50": f, "delta_p50": f, "decision": "keep|revert|veto_p50|pruned|halt", "reason": "one sentence", "chain": [ids of currently kept steps]}. fold_clients makes fold-composition diagnostics free (which clients drive variance) without a separate logging method.

**Why:** The ledger is the harness's memory: it feeds the retry-after-model-switch queue, the leave-one-step-out prune, the null-probe calibration audit, and the final writeup, and JSONL survives crashes mid-loop. Logging per-fold arrays (not just means) is what makes every later re-analysis possible without re-running.

**Leakage:** none

**Accept:** Not a toggle; fixed protocol infrastructure.

### stopping_rule_and_budget (priority: high)

**How:** Three nested stops: (1) STAGE stop — a stage ends when its pre-registered candidate list is exhausted (candidates are enumerated up front by the other dimensions; the harness never invents candidates mid-run). (2) LOOP stop — after stage 5 plus the single conditional reloop, the greedy phase ends unconditionally; additionally end the reloop early if 8 consecutive retried candidates are rejected. (3) HARD BUDGET — max 80 total candidate evaluations (~80 x 15 fits; hours-safe even if HistGB is the base model by then); on hitting it, skip remaining candidates, log them as "decision": "skipped_budget", proceed to re-verification.

**Why:** Open-ended greedy loops converge to overfitting the fold set, not to truth; a pre-registered candidate list plus a hard cap keeps the number of adaptive decisions (and hence CV-selection bias) bounded and known, which also makes the lockbox shrinkage interpretable.

**Leakage:** Indirect: every extra adaptive decision is a small leak of fold information into the config; the budget bounds it.

**Accept:** Not a toggle; fixed protocol.

### pooled_vs_perfold_auc_dual_logging (priority: medium)

**How:** For every candidate also compute the pooled AUC: concatenate (y_te, proba_te) across the 5 folds of each seed, roc_auc_score on the concatenation, average over the 3 seeds; log as "pooled_auc" alongside mean-per-fold. Decisions use ONLY mean-per-fold (per-fold respects the grouped structure; pooled mixes clients and can be moved by between-client score-scale differences rather than within-population ranking). One extra roc_auc_score call, zero extra fits.

**Why:** The two can diverge when a step changes probability calibration across clients more than ranking (e.g., after a model swap or CalibratedClassifierCV step); a large divergence in the ledger is a free diagnostic that a step is reshuffling clients rather than pages. Also, pooled AUC is what a single production ranking across all clients would experience, so reporting both is honest.

**Leakage:** none

**Accept:** Logging-only; never gates. Flag (reason field) any candidate where sign(pooled delta) != sign(per-fold delta).

