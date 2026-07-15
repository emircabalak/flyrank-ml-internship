### precision_at_k_suite (priority: high)

**How:** numpy, ~5 lines: def p_at_k(y, s, k): return y[np.argsort(-s)[:k]].mean(). Compute on each held-out fold for K in {25,50,100,250}; report mean±std across folds plus lift@K = P@K/0.598. Headline stays K=50: it matches the existing W5 metric (0.84) and a plausible refresh-team review batch; K=25 is too granular (one row = 4pp swing), K=250 converges toward the 0.598 base rate so it mostly measures base rate, not model. Show the K-sweep as a small table so the 0.84 is seen in context.

**Why:** P@50 is the stated secondary metric; the sweep shows how sensitive the headline number is to K and anchors it against base rate.

**Leakage:** none — uses held-out labels and scores only.

**Accept:** diagnostic, always keep; P@50 additionally serves as tie-breaker in the harness keep rule when AUC delta is ~0.

### pr_auc_and_roc_curve_report (priority: medium)

**How:** sklearn.metrics.average_precision_score and roc_auc_score per fold; ROC curve via sklearn.metrics.roc_curve (mean curve across folds; RocCurveDisplay if matplotlib present, else skip plot). Report AP alongside its random baseline = 0.598. Honest framing: positives are the MAJORITY class, so the usual 'PR-AUC is better for imbalance' argument does not apply — a random scorer gets AP≈0.60 and PR-AUC will look flatteringly high. Optionally also report AP with labels flipped (minority 'not declining' as positive) as the harder view. ROC-AUC remains primary.

**Why:** standard companion metric; the honest note prevents anyone quoting an inflated PR-AUC as if it were impressive.

**Leakage:** none.

**Accept:** diagnostic, always keep; never used for keep/revert decisions.

### lift_gain_curves (priority: medium)

**How:** no sklearn API — ~10 lines numpy: sort y by score desc, gain = np.cumsum(y_sorted)/y.sum() vs targeted fraction; lift = gain/fraction. Plot per fold plus mean if matplotlib available; always emit the fallback pandas decile table: bin OOF rank into 10 deciles (pd.qcut on rank), positive rate + lift per decile.

**Why:** translates 0.60 AUC into operational terms (top decile is X times better than random targeting) for the public repo readme; expect modest lift (~1.2-1.4x top decile), which is the honest story.

**Leakage:** none.

**Accept:** diagnostic, always keep.

### per_client_auc_distribution (priority: high)

**How:** get out-of-fold scores for every row: cross_val_predict(pipeline, X, y, cv=GroupKFold(5), groups=client_id, method='predict_proba', n_jobs=1) — or LeaveOneGroupOut (32 LogReg fits, still fast) for exactly-one-score-per-client cleanliness. Then df.groupby('client_id').apply(g -> roc_auc_score(g.y, g.s) if g.y.nunique()==2 else np.nan). Report: sorted table (client, n, base rate, AUC), min/median/max, count of degenerate single-class clients, name worst-3 clients, boxplot if matplotlib present.

**Why:** pooled 0.601 can hide 'works on 3 big clients, coin-flip elsewhere'; with 32 clients this is nearly free and is the single most informative reliability view.

**Leakage:** none — strictly out-of-fold predictions.

**Accept:** diagnostic, always keep; if median per-client AUC diverges badly from pooled AUC, switch harness accept metric to median per-client AUC.

### bootstrap_ci_cluster_and_row (priority: high)

**How:** on the held-out set, two recipes, B=2000, numpy only. (a) CLUSTER bootstrap — the correct one under grouped eval: rng.choice(test_clients, size=n_test_clients, replace=True), concat those clients' rows, recompute AUC and pooled P@50 per replicate, percentile [2.5, 97.5]. (b) row bootstrap as secondary, explicitly labeled 'understates uncertainty: rows within a client are correlated'. Answer to the recipe question: resample CLIENTS, because the inference target is generalization to new clients and the grouped split makes client the exchangeable unit; with only ~10 test clients the cluster CI will be wide and lumpy — report that honestly rather than hiding it with the tighter row CI. P@50 note: recompute the pooled top-50 inside each replicate, do not freeze the original top-50 set.

**Why:** 0.601 vs 0.607 is meaningless without a CI; expect the cluster CI to be embarrassingly wide (~±0.03-0.05), which correctly disciplines every downstream claim.

**Leakage:** none.

**Accept:** diagnostic, always keep; CI width calibrates the harness keep threshold — deltas well inside the CI are noise.

### paired_delta_accept_rule (priority: high)

**How:** the harness keep/revert mechanism itself: fix ONE GroupKFold(5) split object (same fold assignment for every toggle), compute per-fold AUC for baseline and toggled config, take per-fold paired deltas. KEEP iff mean delta >= +0.003 AND delta > 0 in >= 4/5 folds; else REVERT. Optional confirmation for borderline keeps: paired cluster bootstrap of the delta (resample clients, score BOTH models on the same replicate, CI on delta). Same-fold pairing is the cheap sklearn-only substitute for a DeLong test.

**Why:** with a ~0.60 ceiling and noisy grouped folds, unpaired mean comparison will greedily keep noise; pairing on identical folds is the biggest single guard against harness churn.

**Leakage:** none.

**Accept:** this IS the accept mechanism; always keep.

### calibration_analysis (priority: medium)

**How:** sklearn.metrics.brier_score_loss on OOF predictions + reliability curve via sklearn.calibration.calibration_curve(y, s, n_bins=10, strategy='quantile'). Report Brier skill vs the constant-0.598 predictor (reference Brier = 0.598*0.402 = 0.2404): BSS = 1 - Brier/0.2404. State explicitly in the notebook: Platt/isotonic (CalibratedClassifierCV) is monotone, so it CANNOT change ROC-AUC or P@K ordering — calibration is report-only unless probabilities are consumed downstream (e.g. shown to clients as risk %).

**Why:** prevents the harness wasting a toggle expecting AUC from calibration, and documents whether the 'probability of decline' numbers are trustworthy as probabilities.

**Leakage:** report-only on OOF preds: none. If CalibratedClassifierCV is ever fitted, it must fit inside training folds only (cv=GroupKFold on train side).

**Accept:** diagnostic, always keep; if CalibratedClassifierCV is tried as a toggle it will show ~0 AUC delta and be reverted by rule — expected and fine.

### score_distribution_overlap (priority: medium)

**How:** OOF predicted probabilities split by true class: overlaid histograms (np.histogram, 30 shared bins; matplotlib if present, else print binned counts table). Numeric overlap stats in pure numpy: mean score per class, KS statistic = max|ECDF_pos - ECDF_neg| via np.searchsorted (no scipy needed). Caption ties it to AUC: at AUC 0.60 the distributions mostly overlap with a small shift — this is the visual proof of the low ceiling.

**Why:** makes the honest-ceiling claim visible to non-ML readers of the public repo; guards against overclaiming.

**Leakage:** none.

**Accept:** diagnostic, always keep.

### permutation_importance_heldout (priority: high)

**How:** sklearn.inspection.permutation_importance(fitted_full_pipeline, X_test_raw_df, y_test, scoring='roc_auc', n_repeats=10, n_jobs=1, random_state=0) per held-out fold, passing the whole Pipeline so permutation happens at raw-column level (preprocessing stays inside — importances are per original column, not per one-hot dummy). Aggregate mean±std across folds; report top-10 plus every feature whose importance interval crosses 0 (candidates for a drop-toggle).

**Why:** model-agnostic (works unchanged if HistGradientBoostingClassifier gets kept), held-out so unbiased; directly ranks which feature-engineering toggles are worth trying next.

**Leakage:** none — permutes allowed columns on held-out data only.

**Accept:** diagnostic, always keep.

### coefficient_report (priority: medium)

**How:** for the LogReg pipeline: names = pipeline[:-1].get_feature_names_out(); coefs = pipeline[-1].coef_[0]; StandardScaler upstream makes coefficients per-SD comparable. Table sorted by |coef| with odds ratio np.exp(coef); average coef across CV folds with std to show sign stability; sanity-check signs against domain intuition (e.g. days_since_last_update should be positive: staler -> more likely declining). Flag any sign flips across folds.

**Why:** free and exact for the linear model; sign-flip instability across folds is a cheap multicollinearity/noise detector.

**Leakage:** none.

**Accept:** diagnostic, always keep while model is linear; drop automatically if final model is HGB (use permutation importance + PDP instead).

### partial_dependence_pure_sklearn (priority: low)

**How:** sklearn.inspection.PartialDependenceDisplay.from_estimator(final_pipeline, X_test, features=['days_since_last_update','content_age_days','impressions_prev_30d','word_count'], kind='both', n_jobs=1) for PD+ICE. Needs matplotlib; numeric fallback: sklearn.inspection.partial_dependence returning grids as a table.

**Why:** SHAP-free shape inspection. Honest note: for LogReg the curves are trivially monotone straight lines in link space — low value now; becomes the main interpretation tool only if the HGB toggle is kept. Run once on the FINAL kept model, not per toggle.

**Leakage:** none.

**Accept:** diagnostic, keep as final-notebook-only step (skip inside the harness loop to save time).

### error_slicing_segments (priority: high)

**How:** on OOF predictions, for each slicing column — content_type, main_intent, competition_level, age_tier, freshness_tier, plus null-pattern flags (word_count_missing, search_volume_missing, main_intent_missing, provider_used_missing) — one pandas groupby table: n, base rate, AUC (skip single-class segments), mean score. Then a combined 'worst slices' table: segments with n>=300 and AUC < pooled_AUC - 0.05. Also crosstab null-pattern x label to check whether missingness itself carries signal (MNAR).

**Why:** finds where the model fails and whether the missing-indicator features are doing real work; null-pattern x label crosstab doubles as a data-quality/MNAR audit.

**Leakage:** none — slicing columns are allowed features or their missingness masks.

**Accept:** diagnostic, always keep.

### comparison_table_baselines (priority: high)

**How:** one pandas table, every row evaluated on the IDENTICAL fixed GroupKFold(5) splits and population (impressions_90d>=100, n=22006 — filter only, never a feature), metrics = mean±std AUC and P@50: (a) DummyClassifier(strategy='prior'); (b) staleness floor: score = raw days_since_last_update column fed to roc_auc_score (no model); (c) prev-traffic floor: score = -impressions_prev_30d; (d) Week-4 CTR-fix baseline RE-RUN on these splits (never paste its old number — different split invalidates comparison); (e) current W5 LogReg baseline; (f) each kept harness toggle cumulatively. Caption states split protocol and population filter.

**Why:** 0.601 means nothing without floors; if the staleness single-column floor scores ~0.57, the model's marginal value is the honest headline. This table IS the deliverable.

**Leakage:** floors use allowed columns only; impressions_90d appears solely as the population filter.

**Accept:** mandatory, always keep.

### seed_stability_report (priority: high)

**How:** rerun the final kept config under GroupShuffleSplit(test_size=0.3, random_state=s) for s in {0,1,2} (GroupKFold in sklearn 1.4/1.5 has no shuffle param, so vary via GroupShuffleSplit); model random_state fixed. Report AUC and P@50 as mean±std across the 3 seeds, alongside the 5-fold GroupKFold fold-level std. Hard rule printed in the notebook: if std across seeds exceeds the claimed improvement over baseline, the improvement is not claimable — say 'directionally positive, within split noise'.

**Why:** with 32 clients, which ~10 land in test dominates the score; quantifies split luck so nobody quotes the best seed.

**Leakage:** none.

**Accept:** mandatory for final reporting, always keep; all public numbers quoted as mean±std.

### topk_stability_and_composition (priority: medium)

**How:** from the 3 seed runs: pairwise Jaccard of pooled top-50 sets restricted to the intersection of test rows (np.intersect1d on page ids / lengths); report mean Jaccard. Composition: value_counts of client_id inside top-50 — flag if <=2 clients supply >50% of it. Complementary per-client variant: precision of top-2-pages-per-client (or top-ceil(50/n_clients)) so pooled P@50 is not conflating client traffic size with decline risk.

**Why:** P@50=0.84 is fragile if the top-50 churns across seeds or is one client's pages; the per-client variant is what an account manager actually consumes.

**Leakage:** none.

**Accept:** diagnostic, always keep.

### below_filter_generalization_check (priority: low)

**How:** train exactly as usual on the filtered population; additionally score the 7994 below-filter rows (they have labels) and report AUC + base rate there, in a clearly-labeled 'out-of-population sanity check — NOT a headline metric' cell. Do not train on these rows (that changes the population definition; if desired that is a separate population toggle, not this method).

**Why:** cheap pseudo-external validation: if AUC roughly holds, the signal is structural; if it collapses, the model leans on population-correlated quirks. Expect degradation — base rate and noise differ below the filter — which is fine to report.

**Leakage:** impressions_90d used only to define which rows are scored, never as a feature.

**Accept:** diagnostic report-only, always keep.

### confidently_wrong_inspection (priority: medium)

**How:** from OOF predictions: table of top-30 highest-score true-negatives (predicted declining, was not) and 30 lowest-score true-positives, showing client_id, content_type, main_intent, content_age_days, days_since_last_update, impressions_prev_30d, word_count, score. Pure pandas sort_values + head. Eyeball for shared patterns (e.g. all FPs from one client, or all missing word_count).

**Why:** cheapest route to new feature ideas at a 0.60 ceiling; there will be many confident errors and their common traits are actionable.

**Leakage:** inspection only; any feature idea it spawns must be re-checked against the forbidden-column list before entering the harness.

**Accept:** diagnostic, always keep.

### learning_curve_sample_sufficiency (priority: low)

**How:** sklearn.model_selection.learning_curve(pipeline, X, y, cv=GroupKFold(5), groups=client_id, train_sizes=np.linspace(0.2,1.0,5), scoring='roc_auc', n_jobs=1, shuffle=True, random_state=0). Plot or table train vs validation AUC by train size.

**Why:** distinguishes 'more data would help' from 'ceiling reached': expect a flat validation curve near 0.60 with small train-val gap, which is direct evidence for the honest-ceiling narrative. Likely confirms the ceiling rather than revealing headroom.

**Leakage:** none — grouped CV inside learning_curve keeps clients unsplit.

**Accept:** diagnostic, run once for the final notebook, keep.

### honest_language_reporting_checklist (priority: medium)

**How:** final markdown cell + optional automated grep of notebook prose (re.findall over the .ipynb markdown cells) enforcing: verbs limited to observed/measured/associated/directional — banned: causes, drives, proves, guarantees, 'significant' without a test; every quoted metric carries split protocol, n, population filter, and ±std or CI; ceiling explanation stated (all label-window features excluded by design, so ~0.60 AUC is expected); P@50 caveated as pooled-across-clients; clients referred to only by anonymized ids in the public repo; no best-seed cherry-picking (mean±std only).

**Why:** explicit requirement for the public repo; a 0.60-AUC model overclaimed is worse than no model.

**Leakage:** n/a.

**Accept:** mandatory, always keep.

