### greedy_single_feature_ablation (priority: high)

**How:** The harness's native move, run as a sweep. Feature units = raw input columns treated atomically (a categorical column + its one-hot block + its missing indicator count as ONE unit). For each of the ~15 units in the current config: rebuild the ColumnTransformer without it, rerun the same GroupKFold(5, groups=client_id) cross_val_score(scoring='roc_auc', n_jobs=1), record delta. Drop the unit with the best positive delta, then repeat the sweep until no drop improves. Candidate order by suspicion: provider_used (21438/22006 null), cpc, competition, search_volume, has_word_count indicator, freshness_tier. ~15 units x 5 folds x ~2s logreg = 3-5 min per sweep pass.

**Why:** The most trustworthy method here: it measures exactly the harness's accept criterion, handles one-hot blocks correctly, and null-dominated features like provider_used are plausible pure-noise carriers. At 25 features L2 usually neutralizes noise, but a 96%-null column filled with 0 plus its indicator can still add fold variance across 32 clients.

**Leakage:** None. Every evaluation is a full grouped CV; nothing is fit outside training folds.

**Accept:** Drop a unit permanently if mean grouped CV AUC delta >= +0.002 and P@50 not worse by more than 0.02; ties (|delta| < 0.002) also drop, fewer features wins.

### poly_interactions_with_univariate_pruning (priority: high)

**How:** The one regime where selection genuinely matters. Pipeline: existing prep (impute/indicators/log1p/one-hot) -> PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) -> VarianceThreshold(0.0) -> SelectKBest(f_classif, k=K) -> StandardScaler() -> LogisticRegression(class_weight='balanced', max_iter=2000). ~25 base cols -> ~325 columns; grid K in {25, 40, 60, 100} x C in {0.1, 1.0}. Optional better-aligned ranker: custom score_func computing per-column |roc_auc_score(y, X[:,j]) - 0.5| (return (scores, zeros) tuple) instead of f_classif, since it matches the target metric and is monotone-transform robust.

**Why:** At 325 features on 22k rows with 32-client group structure, plain L2 overfits fold noise and selection pays for itself. If any interaction carries real signal (e.g. prev_ctr x content_type, days_since_last_update x age_tier), this is the realistic path to +0.005..+0.015. Also the only entry here that can RAISE the ceiling rather than trim variance.

**Leakage:** SelectKBest MUST sit inside the Pipeline so it refits on each training fold; fitting it once on all rows before CV inflates AUC (selection sees test-client labels). No groups needed by the selector itself.

**Accept:** Keep the (K, C) cell if mean grouped CV AUC delta >= +0.003 over baseline (slightly higher bar since the grid has more cells to get lucky in); revert otherwise.

### permutation_importance_pruning (priority: high)

**How:** Per outer fold: split the training fold with GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0) into sub-train/val; fit the full baseline pipeline on sub-train; r = sklearn.inspection.permutation_importance(pipe, X_val_raw, y_val, scoring='roc_auc', n_repeats=10, random_state=0, n_jobs=1). Pass the PIPELINE and RAW columns so permuting a categorical column shuffles its whole one-hot block at once. Drop every raw column with r.importances_mean <= 0, refit on the full training fold, score the test fold. Threshold grid: {<=0, <= -0.001}.

**Why:** Model-specific and directly AUC-denominated, so it catches features that actively hurt (again provider_used, cpc are candidates). Caveat: correlated pairs (word_count/char_count, the three prev-30d counts) split their importance and can BOTH read as ~0, so it may over-drop; check drops against the ablation results.

**Leakage:** Importance is computed on a grouped validation split inside the training fold only; never on outer test clients. Using a non-grouped inner split would rank client-memorizing features too high, so keep GroupShuffleSplit.

**Accept:** Keep if mean grouped CV AUC delta >= +0.002; also sanity-check the dropped set is stable across outer folds (if each fold drops a different set, revert regardless of delta).

### correlation_pruning (priority: high)

**How:** Custom ~15-line transformer (BaseEstimator, TransformerMixin), placed after imputation/log1p, before scaler, numeric columns only. fit: c = np.abs(np.corrcoef(X, rowvar=False)); np.fill_diagonal(c, 0); iterate upper triangle, for each pair with c > thr drop the member with the higher null-rate (or the later column); store self.keep_ indices. transform: X[:, self.keep_]. Grid thr in {0.85, 0.9, 0.95}.

**Why:** word_count/char_count are near-duplicates (r ~ 0.99) and impressions/clicks/sessions_prev_30d are strongly inter-correlated. L2 already stabilizes collinear coefficients, so expected delta is small, but this is one of the few structural redundancies actually present in this pool and it is nearly free to test.

**Leakage:** None (uses X only, no y). Still fit it inside the Pipeline per fold for strictness.

**Accept:** Keep if delta >= +0.002, or if delta >= -0.001 with >= 2 columns removed (simplicity tie-break).

### sequential_forward_selection_grouped (priority: medium)

**How:** Per outer fold, precompute inner grouped splits and hand them to SFS as a materialized list (this is how you make its internal CV group-aware without metadata routing): inner = list(GroupKFold(3).split(X_tr, y_tr, groups_tr)); sel = SequentialFeatureSelector(LogisticRegression(class_weight='balanced', max_iter=1000), direction='forward', n_features_to_select='auto', tol=1e-3, scoring='roc_auc', cv=inner, n_jobs=1). Run on the post-prep scaled matrix (~25 cols). ~25x25x3 ~ 2000 logreg fits per outer fold, each <0.1s -> a few minutes total.

**Why:** The wrapper method best aligned with the objective: it greedily optimizes grouped AUC itself. Plausible outcome is that only ~6-8 features carry all the signal and the rest are variance; small but real chance of a gain via variance reduction. More likely neutral at 25 features.

**Leakage:** Two traps: (1) default cv=5 is StratifiedKFold and splits WITHIN clients, selecting client-memorizing features -- always pass the precomputed GroupKFold splits; (2) run selection per outer training fold, never once on all data. Note the outer metric stays honest even with a bad inner cv; only selection quality suffers.

**Accept:** Keep if mean grouped CV AUC delta >= +0.002; on ties prefer the selected subset if it has <= half the features.

### stability_selection_client_subsample (priority: medium)

**How:** Manual, ~25 lines, no extra deps. On each outer training fold (scaled matrix): counts = zeros(p); clients = np.unique(g_tr); for seed in range(30): pick len(clients)//2 clients via np.random.RandomState(seed).choice(replace=False), mask rows, fit LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced'), counts += (|coef_|>1e-8). keep = counts/30 >= freq. Grid: C in {0.05, 0.1}, freq in {0.6, 0.7, 0.8}. 30 liblinear fits on ~10k rows = seconds.

**Why:** Uniquely well-matched to this problem's failure mode: with 32 clients dominating the variance, features whose L1 selection survives random client subsets are exactly the ones that transfer to unseen clients. Modest gain possible; at worst identifies the same stable core the other methods find.

**Leakage:** Subsample at CLIENT level, not row level -- row bootstrap keeps every client in every replicate and defeats the purpose. Run entirely within each outer training fold.

**Accept:** Keep if delta >= +0.002; also report the stable feature set as a cross-check for ablation/permutation drops even if reverted.

### select_from_model_l1_logreg (priority: medium)

**How:** After StandardScaler: SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=c, class_weight='balanced'), threshold=1e-5), c in {0.01, 0.05, 0.1}, followed by the usual L2 LogisticRegression. Inside the Pipeline. Also valid as the pruner in the poly-interactions pipeline (there it competes with SelectKBest and is worth a cell in that grid).

**Why:** Embedded sparsity; at 25 features it is roughly equivalent to just tuning regularization and overlaps the hyperparameter-tuning dimension -- expect neutral on the base config. Its real use is as an alternative pruner after PolynomialFeatures where the 325-column regime gives it something to do.

**Leakage:** Uses y; must be fit inside the Pipeline on training folds only. No groups needed.

**Accept:** Keep if delta >= +0.002; else revert. If neutral on base config, still try it once inside the poly pipeline before discarding the idea.

### univariate_f_classif_percentile (priority: medium)

**How:** SelectPercentile(f_classif, percentile=p), p in {50, 70, 85}, inside the Pipeline after prep (scale-invariant, position vs scaler irrelevant). Variants for exhaustiveness: SelectFdr(f_classif, alpha=0.05) / SelectFwe for FDR-controlled cutoffs instead of a percentile grid; chi2 is redundant here (needs non-negative input, adds nothing over F-stat on this data) -- skip it.

**Why:** Honest: at ~25 features, filtering by marginal F-stat mostly duplicates what L2 shrinkage already does, and it can drop features that only work in combination. Expect neutral. Included because it costs seconds and is the textbook baseline the fancier methods must beat.

**Leakage:** Uses y; fit inside Pipeline per training fold. Fitting once on all 22k rows before CV is the classic selection-leak -- do not.

**Accept:** Keep best p if delta >= +0.002; expect revert.

### univariate_mutual_info_percentile (priority: low)

**How:** SelectPercentile(score_func=partial(mutual_info_classif, random_state=0, n_neighbors=3), percentile=p), p in {50, 70, 85}, inside Pipeline. MI on 22k x 25 is a few seconds per fold.

**Why:** Honest mismatch: MI credits nonlinear dependence, but the downstream model is linear and cannot exploit it, so MI-selected features may not help logreg at all. Only becomes coherent if the model toggle later switches to HistGradientBoostingClassifier. Expect neutral on the current config.

**Leakage:** Uses y; inside Pipeline only. Set random_state or selection differs across folds for spurious reasons.

**Accept:** Keep if delta >= +0.002; re-try automatically if the model ever becomes tree-based.

### select_from_model_tree_importances (priority: low)

**How:** SelectFromModel(ExtraTreesClassifier(n_estimators=300, max_depth=None, class_weight='balanced', random_state=0, n_jobs=1), threshold=t), t in {'median', 'mean', '0.5*mean'}, inside Pipeline before scaler+logreg. NOTE: HistGradientBoostingClassifier exposes no feature_importances_, so ExtraTrees/RandomForest is the only sklearn option for this. ~20-30s per fold at n_jobs=1.

**Why:** Impurity importances are biased toward high-cardinality numerics and systematically understate one-hot dummies, so the selected set skews numeric. Selecting via a tree for a linear consumer is also a model mismatch. Expect neutral-to-negative; included so the harness can reject it with data.

**Leakage:** Uses y; inside Pipeline per fold. No groups needed.

**Accept:** Keep if delta >= +0.002; expect revert.

### rfe_fixed_k (priority: low)

**How:** RFE(LogisticRegression(class_weight='balanced', max_iter=1000), n_features_to_select=k, step=1) after StandardScaler, k in {10, 15, 20}, inside Pipeline. ~p fits per fold, seconds.

**Why:** Recursive |coef| elimination on standardized features is close to one-shot coefficient pruning, which L2 already approximates. At 25 features expect the k=20 cell to tie baseline and smaller k to lose. Cheap textbook entry the harness can kill quickly.

**Leakage:** Uses y; inside Pipeline per fold. No internal CV, so no group concern.

**Accept:** Keep best k if delta >= +0.002; expect revert.

### rfecv_grouped (priority: low)

**How:** Cannot go inside a plain Pipeline because its internal CV needs groups. Per outer fold: inner = list(GroupKFold(3).split(X_tr, y_tr, groups_tr)); RFECV(LogisticRegression(class_weight='balanced', max_iter=1000), step=1, min_features_to_select=5, scoring='roc_auc', cv=inner, n_jobs=1); take support_, refit final pipe on the reduced set. WARNING (the trap this entry exists to document): default cv=5 uses StratifiedKFold, which splits within clients -- selection then favors client-memorizing features. Always pass the materialized GroupKFold splits. ~1-2 min per outer fold.

**Why:** Auto-k version of RFE; with grouped inner CV it usually concludes 'keep nearly everything' at this dimensionality. Expect neutral. Strictly dominated by SFS-forward here (same cost class, same estimator, better search direction for small signal sets), so run it after SFS only if SFS surprises.

**Leakage:** Inner CV must be group-aware as above; run per outer training fold, never once globally. A non-grouped inner cv does NOT bias the outer metric, but it degrades the selection it produces.

**Accept:** Keep if delta >= +0.002; expect revert.

### sequential_backward_selection_grouped (priority: low)

**How:** Same harness code as SFS-forward with direction='backward', n_features_to_select='auto', tol=-1e-3 (stop when removing any feature costs more than 0.001 AUC), cv=precomputed GroupKFold(3) splits per outer fold, n_jobs=1. More fits than forward when the retained set is large: budget ~5 min.

**Why:** Backward keeps feature interactions intact while pruning, which matters more at high dimension; at 25 features it almost always lands where forward lands at ~2x the cost. Run only if forward showed movement.

**Leakage:** Same as SFS-forward: precomputed grouped inner splits, per outer fold only.

**Accept:** Keep if delta >= +0.002 AND it beats the SFS-forward result; otherwise revert.

### variance_threshold (priority: low)

**How:** VarianceThreshold(threshold=t) inserted after one-hot, before scaler, t in {0.0, 0.005} (on 0/1 dummies, var = p(1-p), so 0.005 removes dummies rarer than ~0.5%). One line in the Pipeline.

**Why:** Pure hygiene: removes fold-constant dummy columns (a category absent from some training fold). StandardScaler already maps zero-variance cols to 0 for logreg, so measurable delta ~ 0.000. Include as a free guard, not as a hoped gain.

**Leakage:** None (no y); fit per fold inside Pipeline.

**Accept:** Keep t=0.0 unconditionally if delta >= -0.0005 (harmless guard); drop the 0.005 cell unless delta >= +0.002.

### pca_replacement (priority: low)

**How:** Pipeline: prep -> StandardScaler -> PCA(n_components=nc, random_state=0) -> LogisticRegression(class_weight='balanced'), nc in {5, 10, 0.90, 0.95} (floats = retained variance fraction). Seconds per fold.

**Why:** Honest expected failure: L2 logreg is essentially rotation-invariant, so PCA can only change things through truncation, and unsupervised directions on mixed one-hot/count features rarely align with the label. On tabular data this loses AUC far more often than it gains. Included so the harness rejects it with a number.

**Leakage:** None (unsupervised); fit inside Pipeline per fold.

**Accept:** Keep if delta >= +0.002; expect clear revert.

### pca_augmentation (priority: low)

**How:** After StandardScaler: FeatureUnion([('id', 'passthrough'), ('pca', PCA(n_components=5, random_state=0))]) -> LogisticRegression. One-line toggle.

**Why:** Honest near-no-op: for a linear model, principal components are exact linear combinations of columns already present, so this adds zero expressiveness and only perturbs the L2 penalty geometry. Expect delta ~ 0.000. Listed for exhaustiveness; would only matter for a tree model downstream (oblique splits).

**Leakage:** None; inside Pipeline.

**Accept:** Keep if delta >= +0.002; expect revert. Re-try once if the model toggle switches to HistGradientBoostingClassifier.

### truncated_svd_after_poly (priority: low)

**How:** Only meaningful in the wide regime: swap the SelectKBest step of the poly-interactions pipeline for TruncatedSVD(n_components=k, random_state=0), k in {10, 20, 40}, then scaler -> logreg. On the base 25-column dense matrix TruncatedSVD is just uncentered PCA -- skip that cell entirely.

**Why:** Competes with univariate pruning as the dimensionality tamer for 325 poly columns. Unsupervised compression usually loses to supervised selection when signal is sparse (a few good interactions among many junk ones), which is the likely situation here. Expect it to lose to SelectKBest.

**Leakage:** None (unsupervised); inside Pipeline.

**Accept:** Keep only if it beats both baseline (+0.002) and the SelectKBest poly variant.

### feature_agglomeration (priority: low)

**How:** After StandardScaler: FeatureAgglomeration(n_clusters=k, linkage='ward'), k in {8, 12, 16} -> LogisticRegression. Seconds per fold.

**Why:** Merges correlated features into cluster means -- the soft version of correlation pruning. At 25 features the information destroyed by averaging (e.g. pooling cpc with search_volume) likely exceeds the variance saved. Expect neutral-to-negative; corr-pruning is the better-targeted sibling.

**Leakage:** None (no y); inside Pipeline per fold.

**Accept:** Keep if delta >= +0.002 AND it beats correlation pruning; otherwise revert.

### manual_shortlist_floor (priority: medium)

**How:** Diagnostic toggle, one config: restrict features to [log1p(impressions_prev_30d), log1p(clicks_prev_30d), log1p(sessions_prev_30d), prev_ctr = clicks_prev_30d/np.maximum(impressions_prev_30d,1), days_since_last_update] -> scaler -> logreg. One CV run, ~10s.

**Why:** Not expected to win -- it exists to answer 'do the static page properties carry ANY transferable signal?'. If this 5-feature model ties 0.601, every selection method above should converge to roughly this set and the static-feature engineering budget should be cut. If it clearly loses, the static pool is worth keeping and selection should be conservative.

**Leakage:** None; prev-window and static columns only, standard grouped CV.

**Accept:** Keep only if delta >= 0.000 (simplicity wins exact ties); primarily read it as a diagnostic, not a candidate.

