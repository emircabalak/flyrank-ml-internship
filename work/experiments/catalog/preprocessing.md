### median_imputation_numerics (priority: high)

**How:** Replace baseline fillna(0) with SimpleImputer(strategy='median') on the numeric block (search_volume, competition, cpc, word_count, char_count, content_age_days, days_since_last_update, 3 prev-30d counts) inside the Pipeline/ColumnTransformer so it fits on the train fold only. Keep the existing has_keyword/has_word_count indicators so information about missingness is not lost.

**Why:** fillna(0) puts nulls at the extreme low end of skewed columns like search_volume and cpc, distorting the linear fit; median is the standard fix and costs nothing. Realistic small AUC gain for LR.

**Leakage:** None if fit inside the pipeline on train folds. Pre-computing the median on the full 22k rows would leak test-client distributions; avoid by using SimpleImputer in the Pipeline.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002 over baseline; revert otherwise.

### mean_imputation_numerics (priority: low)

**How:** SimpleImputer(strategy='mean') on the same numeric block, same pipeline placement as median variant. Toggle is mutually exclusive with median_imputation_numerics; harness tries each separately.

**Why:** Cheap bakeoff arm. On heavily skewed columns mean is usually worse than median; included so the harness can reject it empirically.

**Leakage:** None when fit on train fold inside the pipeline.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise (expected revert).

### universal_missing_indicators (priority: medium)

**How:** SimpleImputer(strategy='median', add_indicator=True) so every numeric column with nulls gets a binary indicator, replacing the hand-rolled has_keyword/has_word_count pair. Equivalent explicit form: sklearn.impute.MissingIndicator(features='missing-only') in a FeatureUnion.

**Why:** Missingness is plausibly informative (no keyword data, no word count = never crawled/parsed). Baseline only flags two families; this covers all of them including days_since_last_update if it has nulls.

**Leakage:** None; indicators are functions of the row itself.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise.

### collapse_null_family_indicators (priority: medium)

**How:** The null counts show search_volume/competition/cpc share 2468 nulls (one 'no keyword data' event) and word_count/char_count/tiers share 7699 (one 'no content parse' event). Verify overlap once with df[cols].isna().all(axis=1).sum() == df[cols].isna().any(axis=1).sum(), then emit exactly two binary features no_keyword_data and no_content_stats instead of per-column indicators. Pure row-wise numpy: df[cols].isna().all(axis=1).astype(int).

**Why:** Removes redundant collinear indicator columns and encodes the real underlying event; mostly a cleanliness win for LR coefficients, small or zero AUC effect.

**Leakage:** None; row-wise transformation.

**Accept:** Keep if AUC delta >= 0 (it simplifies the feature set); revert only if AUC drops > 0.002.

### per_category_median_imputation (priority: low)

**How:** Custom transformer (BaseEstimator+TransformerMixin): fit() computes train-fold median of word_count and char_count grouped by content_type via df.groupby('content_type')[col].median(), plus a global-median fallback for unseen/null categories; transform() fills nulls with the mapped value. Same idea for search_volume grouped by main_intent as a second variant.

**Why:** Word counts plausibly differ by content type (blog vs product page), so conditional medians are less distorting than a global one. Gain over global median is usually marginal; cheap to test.

**Leakage:** Group medians MUST be computed on the train fold inside the pipeline. Computing them on the full frame leaks test-client content distributions (still not label leakage, but breaks the grouped-eval contract).

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002 vs the best global-imputation arm; revert otherwise.

### knn_imputer (priority: low)

**How:** Pipeline: StandardScaler -> KNNImputer(n_neighbors=k, weights='distance') on the numeric block, k grid [5, 15, 31]. Note KNNImputer requires the scaler first or distances are dominated by char_count. ~22k x 22k chunked distance computation: runs in low minutes, acceptable but the slowest imputer arm.

**Why:** Nulls here are structural (whole families missing together), not random, so KNN has little signal to exploit; expected to match median at best. Included for completeness.

**Leakage:** None if fit on train fold inside the pipeline; never fit on the full frame.

**Accept:** Keep best-k only if mean GroupKFold(5) AUC delta >= +0.003 (higher bar to justify runtime); revert otherwise (expected revert).

### iterative_imputer (priority: low)

**How:** from sklearn.experimental import enable_iterative_imputer; IterativeImputer(estimator=None (BayesianRidge default), max_iter=10, random_state=0, n_jobs is not a param so no Windows issue) on the numeric block inside the pipeline.

**Why:** Same structural-missingness caveat as KNN: when search_volume/competition/cpc are all null together there are few informative predictors left to regress on. Expected no gain; cheap enough to let the harness reject.

**Leakage:** None if fit on train fold inside the pipeline.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.003; revert otherwise (expected revert).

### missing_as_category_all_cats (priority: high)

**How:** For provider_used (71% null), model_used (5733 null), main_intent (2374 null), competition_level (2610 null): df[col] = df[col].fillna('__missing__') applied row-wise before OneHotEncoder(handle_unknown='ignore'). This also pulls provider_used and model_used INTO the feature set (baseline currently omits them).

**Why:** The null pattern is the signal: provider_used null very likely means the page was never AI-refreshed, which is directly related to content staleness and decline. Making NaN an explicit level is the only sane encoding at 71% null. One of the most promising items in this dimension.

**Leakage:** None. fillna with a constant is row-wise; verify provider_used/model_used describe actions taken BEFORE the label window (they are in the allowed pool per spec, so treated as safe).

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise.

### has_ai_refresh_binary (priority: high)

**How:** Single binary feature df['provider_used'].notna().astype(int) (optionally also model_used.notna()), used INSTEAD of one-hot levels of provider_used. Toggle against missing_as_category_all_cats to see whether the 2 provider levels add anything beyond the presence flag.

**Why:** With 71% null and only 2 non-null categories, most of the information is in presence/absence; the binary is the lower-variance encoding and may generalize better across clients (some clients may exclusively use one provider, making provider levels a proxy for client identity).

**Leakage:** The provider LEVELS risk acting as client fingerprints under grouped CV (provider choice is likely per-client), which inflates within-fold fit but hurts held-out clients; the binary flag reduces that. No label leakage either way.

**Accept:** Keep whichever of {binary flag, full missing-as-category OHE} has higher mean GroupKFold(5) AUC; revert if neither beats baseline by +0.002.

### scaler_robust (priority: medium)

**How:** Swap StandardScaler -> RobustScaler(quantile_range=(25.0, 75.0)) in the numeric branch.

**Why:** Numeric block has heavy-tailed counts; robust scaling reduces the leverage of extreme pages on LR. Modest realistic gain.

**Leakage:** None; fit on train fold inside pipeline.

**Accept:** Part of a scaler bakeoff: keep the single best scaler by mean GroupKFold(5) AUC if it beats StandardScaler by >= +0.002; else keep StandardScaler.

### scaler_minmax (priority: low)

**How:** Swap StandardScaler -> MinMaxScaler().

**Why:** Bakeoff arm; usually worse than Standard/Robust for LR on heavy tails because one outlier compresses everything else. Included for completeness.

**Leakage:** None; fit on train fold.

**Accept:** Scaler bakeoff rule above (expected revert).

### scaler_maxabs (priority: low)

**How:** Swap StandardScaler -> MaxAbsScaler().

**Why:** Bakeoff arm; same tail-compression weakness as MinMax on this data. Expected revert.

**Leakage:** None; fit on train fold.

**Accept:** Scaler bakeoff rule above (expected revert).

### scaler_quantile_normal (priority: high)

**How:** Swap StandardScaler -> QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, n_train_fold), subsample=None, random_state=0).

**Why:** Rank-gaussianizes the skewed counts, which often beats log1p+Standard for LR on web-traffic-style data. One of the strongest scaler arms; note it partially subsumes the log1p steps (test with and without them).

**Leakage:** None; quantiles fit on train fold inside pipeline. Do NOT fit on the full frame.

**Accept:** Scaler bakeoff rule: keep if mean GroupKFold(5) AUC beats StandardScaler arm by >= +0.002.

### scaler_quantile_uniform (priority: medium)

**How:** QuantileTransformer(output_distribution='uniform', n_quantiles=min(1000, n_train_fold), random_state=0).

**Why:** Bakeoff arm; uniform output is fine for LR too (bounded, rank-based). Usually close to the normal variant.

**Leakage:** None; fit on train fold.

**Accept:** Scaler bakeoff rule above.

### scaler_power_yeojohnson (priority: medium)

**How:** PowerTransformer(method='yeo-johnson', standardize=True). Handles zeros/negatives so it works on fillna(0) columns directly.

**Why:** Parametric alternative to quantile transform; sometimes better when the skew is genuinely log-normal-ish, sometimes unstable on spike-at-zero columns. Worth one arm.

**Leakage:** None; fit on train fold.

**Accept:** Scaler bakeoff rule above.

### no_scaling_passthrough (priority: low)

**How:** Replace the scaler with 'passthrough' in the ColumnTransformer. ONLY meaningful when the model dimension has switched to HistGradientBoostingClassifier; for LogisticRegression keep a scaler.

**Why:** Trees are scale-invariant; dropping the scaler removes a no-op and lets HGB see raw values with its native binning. Zero expected AUC change, pure simplification, conditional on model choice.

**Leakage:** None.

**Accept:** Only toggled if current best model is tree-based: keep if AUC delta >= -0.001 (i.e., not worse), since it simplifies the pipeline.

### winsorize_clip_percentiles (priority: medium)

**How:** Custom transformer: fit() stores np.nanpercentile(X_train, [lo, hi], axis=0); transform() applies np.clip per column. Grid over (lo, hi) in {(0.5, 99.5), (1, 99), (5, 95)}. Apply to raw numerics before scaling. Also clip days_since_last_update and content_age_days at lower bound 0 unconditionally as a sanity guard.

**Why:** Caps the leverage of extreme count/length pages on LR without discarding rows. Cheap and often worth a small bump when StandardScaler is kept; redundant if quantile transform wins the scaler bakeoff.

**Leakage:** Percentiles MUST come from the train fold (fit/transform pattern). np.clip with full-data percentiles would leak test-client distributions.

**Accept:** Keep best grid point if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise. Skip entirely if scaler_quantile_* was kept.

### log1p_extended_skewed (priority: high)

**How:** FunctionTransformer(np.log1p) extended beyond the 3 prev-30d counts to: search_volume, cpc, word_count, char_count, content_age_days, days_since_last_update (all non-negative after imputation). Implement as a ColumnTransformer branch: impute -> log1p -> scaler. Sub-toggle: replace raw prev-30d counts with their log1p versions instead of keeping both (drops 3 collinear columns).

**Why:** Baseline only logs the prev counts; search_volume and cpc are classically log-normal and word/char counts are right-skewed. Direct small gain for LR. Redundant under QuantileTransformer.

**Leakage:** None; elementwise transform.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise. Test the drop-raw sub-toggle only if the main toggle is kept.

### ohe_drop_variants (priority: low)

**How:** OneHotEncoder bakeoff: drop=None (baseline-equivalent) vs drop='first' vs drop='if_binary', always with handle_unknown='ignore' and sparse_output=False.

**Why:** With L2-regularized LR, drop=None is usually fine or slightly better; drop='first' mainly matters for unregularized models. Near-zero expected delta; one-line toggle so let the harness check.

**Leakage:** None; categories fit on train fold. handle_unknown='ignore' is REQUIRED under grouped splits because a held-out client can carry a category level absent from training clients.

**Accept:** Keep best variant if it beats drop=None by >= +0.002; otherwise keep drop=None. handle_unknown='ignore' is kept unconditionally as a crash fix.

### ordinal_tier_encoding (priority: medium)

**How:** Replace OHE with OrdinalEncoder(categories=[explicit_ordered_list_per_col], handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1) for the naturally ordered cats: age_tier (or just use existing age_tier_order numeric), freshness_tier, word_count_tier, char_count_tier, competition_level (low<medium<high). Feed the resulting integers into the numeric scaler branch. Read the actual level strings once from df[col].unique() to build the ordered lists.

**Why:** Encodes monotone structure in 1 column instead of 4, which is a better bias for LR when the effect is monotone in the tier. Note tiers are coarsenings of word_count/char_count/content_age which are already numeric features, so tiers may add nothing at all; that is also worth learning.

**Leakage:** None; ordering is fixed a priori, not learned from labels.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002 vs OHE-of-tiers arm; revert otherwise. Also try a third arm: drop tier columns entirely (they duplicate the raw numerics).

### frequency_encoding (priority: low)

**How:** Custom transformer: fit() stores value_counts(normalize=True) per categorical column from the train fold; transform() maps categories to their train frequency, unseen/null -> 0. Apply to content_type, main_intent, model_used.

**Why:** Frequency encoding shines on high-cardinality columns; every categorical here has 2-5 levels, so it carries strictly less information than OHE. Expected revert; included for completeness.

**Leakage:** Frequencies from train fold only. No label involved, so no label leakage; full-data frequencies would only leak distributional info, still avoid.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise (expected revert).

### oof_target_encoding (priority: low)

**How:** sklearn.preprocessing.TargetEncoder(cv=5, smooth='auto', target_type='binary', random_state=0) on the categorical block, inside the pipeline. IMPORTANT caveat: TargetEncoder's internal cross-fitting uses row-wise KFold, not GroupKFold, so rows of the same client land in different internal folds and a client's own label mean bleeds into its encodings. Safer variant if tested seriously: hand-rolled OOF encoding using GroupKFold(5, groups=client_id) on the train fold to compute per-category smoothed means.

**Why:** Target encoding earns its keep at high cardinality; at 2-5 levels it is mathematically close to what LR already learns from OHE. Expected no gain, plus the group-leakage caveat inflates CV optimistically. Enumerate, expect revert.

**Leakage:** REAL risk: internal KFold not group-aware -> within-client label leakage into encodings -> optimistic CV AUC that will not transfer. Mitigate with the hand-rolled GroupKFold OOF variant, or reject the method. Never target-encode client_id itself (pure leakage, and test clients are unseen).

**Accept:** Only the group-aware hand-rolled variant is eligible to KEEP; sklearn TargetEncoder arm is measurement-only. Keep if group-aware variant beats OHE by >= +0.003; revert otherwise (expected revert).

### rare_category_grouping (priority: low)

**How:** OneHotEncoder(min_frequency=20, handle_unknown='infrequent_if_exist', sparse_output=False); grid min_frequency in {10, 20, 0.01}. Mainly relevant for model_used (5 levels, some possibly tiny).

**Why:** Rare model_used levels one-hot to near-constant columns that fit noise on the few clients that use them. Low expected delta given only ~14 total category levels, but it is a one-parameter toggle.

**Leakage:** Frequencies computed on train fold by the encoder; safe. Also guards against unseen-level fragility across client folds.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.001 or unchanged with fewer columns; revert if worse.

### duplicate_row_drop_train_only (priority: low)

**How:** In the harness's fit step (not the pipeline): mask = X_train.duplicated(keep='first') computed over the model feature columns only; drop those rows (and y) from the train fold. Test fold untouched. Report the duplicate count once with df[feature_cols].duplicated().sum().

**Why:** Anonymized page-level data often contains near-clones (same template pages). Deduping train reweights the effective distribution slightly; effect is usually negligible but the check is 2 lines.

**Leakage:** None: modifying only the training fold is always legitimate under grouped eval. Never drop test rows (that changes the metric population).

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise (expected revert).

### near_zero_variance_drop (priority: low)

**How:** VarianceThreshold(threshold=1e-8) as the last preprocessing step after OHE+scaling. Optionally a stricter arm at threshold=0.01 on the scaled block.

**Why:** Pure hygiene: kills all-constant columns produced by OHE of levels absent in a train fold. AUC-neutral by construction at threshold~0; the stricter arm rarely helps LR with L2.

**Leakage:** None; variances from train fold inside pipeline.

**Accept:** Keep threshold=1e-8 unconditionally as hygiene (delta ~ 0 expected); keep 0.01 arm only if AUC delta >= +0.002.

### correlation_redundancy_pruning (priority: medium)

**How:** One-off analysis then a static drop list: compute X_train[numerics].corr(method='pearson').abs() on one train fold; for pairs > 0.95 drop the member with more nulls. Near-certain hit: char_count vs word_count (drop char_count and char_count_tier). Implement the kept drop list as a plain columns-to-exclude constant, not a fitted step.

**Why:** word_count/char_count are ~perfectly correlated (both null together in the same 7699 rows, both measure length). L2 LR tolerates collinearity, so expect AUC ~ flat, but fewer columns and stabler coefficients; genuinely useful if a later feature-selection step runs.

**Leakage:** None if the correlation analysis uses a train fold; even full-data Pearson among features involves no labels, but stay consistent and use train.

**Accept:** Keep if AUC delta >= -0.001 (accept as simplification when not worse); revert if it costs more than that.

### isolationforest_outlier_flag_feature (priority: medium)

**How:** Custom transformer: fit() trains IsolationForest(n_estimators=100, contamination='auto', random_state=0, n_jobs=1) on the imputed+scaled train-fold numerics; transform() appends decision_function(X) as one extra numeric feature (and optionally the binary predict()==-1 flag).

**Why:** 'This page is weird' is a plausible weak signal for decline (thin/anomalous pages decay). Unsupervised, cheap at 22k rows. Modest expectations.

**Leakage:** None: fit on train fold only, no labels used, test rows are only scored. Fully legitimate under grouped eval.

**Accept:** Keep if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise.

### outlier_removal_train_only (priority: low)

**How:** Harness fit step: fit IsolationForest(contamination=c, random_state=0, n_jobs=1) on train-fold numerics, drop rows with predict()==-1 from the train fold only; grid c in {0.02, 0.05}. Test fold untouched.

**Why:** Removes high-leverage anomalies from the LR fit. Sometimes helps linear models, often does nothing; class balance shifts slightly so class_weight='balanced' recomputes correctly anyway.

**Leakage:** None as long as ONLY the training fold is filtered. Removing test rows would corrupt the metric population and is forbidden.

**Accept:** Keep best c if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise.

### kbins_discretization (priority: medium)

**How:** KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile', subsample=None) on the skewed numerics (search_volume, cpc, prev-30d counts, word_count) as an ALTERNATIVE to scaler+raw, giving LR a piecewise-constant response. Grid n_bins in {4, 5, 8}.

**Why:** Lets LR capture non-monotone effects (e.g., decline risk highest at mid-range prev impressions) without a model change. Borderline feature-engineering, listed here as an encoding arm; overlaps with what HGB would learn natively, so mainly valuable while the model stays linear.

**Leakage:** None; bin edges fit on train fold inside pipeline.

**Accept:** Keep best n_bins if mean GroupKFold(5) AUC delta >= +0.002; revert otherwise. Deprioritize if model dimension switches to HGB.

### impute_zero_vs_median_for_prev_counts (priority: medium)

**How:** Targeted sub-toggle: for impressions_prev_30d/clicks_prev_30d/sessions_prev_30d specifically, keep fillna(0) (a page with no prev-window data plausibly HAD zero traffic, so 0 is the semantically correct value) even if the global imputer moves to median; implement by splitting the ColumnTransformer numeric branch in two.

**Why:** Blanket median imputation can be wrong for count columns where missing means zero. This protects the strongest signal columns (prev-window traffic) from being distorted by the imputation bakeoff.

**Leakage:** None; constant fill is row-wise.

**Accept:** Tested jointly with median_imputation_numerics: keep the split-branch version if it beats all-median by >= +0.001; else keep the simpler single branch.

