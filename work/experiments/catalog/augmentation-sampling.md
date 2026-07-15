### class_weight=None (drop 'balanced') (priority: medium)

**How:** One-line toggle: LogisticRegression(class_weight=None, max_iter=1000). Baseline uses 'balanced', which with base rate 0.598 down-weights the POSITIVE majority (~0.836 vs 1.254). Removing it restores natural weighting.

**Why:** Imbalance is mild and inverted (positives are majority), so 'balanced' was never justified. Class weights mostly shift the intercept for LR but do perturb the coefficient direction slightly; AUC delta will be small either way. Cheap sanity toggle; P@50 may move more than AUC.

**Leakage:** none

**Accept:** Keep if mean grouped-CV AUC delta >= +0.002 with P@50 not worse by more than 0.02; otherwise revert.

### class_weight custom grid (priority: low)

**How:** Grid over class_weight={0:1, 1:w} for w in [0.5, 0.67, 0.836, 1.0, 1.25, 1.5, 2.0] (0.836 = the 'balanced' value). Evaluate every point on the SAME grouped folds; the toggle adopts the single best w.

**Why:** Covers the space between None and balanced and beyond. Honest expectation: near-flat AUC curve because LR ranking is weakly sensitive to class weights at 60/40; picking the max of 7 noisy points risks selection noise, hence the stricter threshold below.

**Leakage:** none

**Accept:** Keep best w only if its mean CV AUC beats baseline by >= +0.003 (stricter because 7 points were tried); tie-break on P@50.

### sample_weight proportional to prev-window traffic (priority: high)

**How:** Inside each train fold: w = np.log1p(X_tr['impressions_prev_30d']); w *= len(w)/w.sum(); pipe.fit(X_tr, y_tr, logisticregression__sample_weight=w). Variant grid: w = log1p, w = sqrt(impressions_prev_30d), w = 1.0 + (impressions_prev_30d >= 100). Weights are never used at predict time.

**Why:** Two real mechanisms: (1) trend labels computed on low-traffic pages are mostly count noise, so down-weighting them cleans the training signal; (2) the eval population is impressions_90d>=100 pages, which correlates strongly with prev-30d traffic, so this aligns the train distribution with the eval population without touching forbidden columns. One of the few sampling ideas with a genuine causal story here.

**Leakage:** None. impressions_prev_30d predates the label window and is already an allowed feature. Compute weights per train fold only; never weight or filter the eval fold.

**Accept:** Keep the best variant if mean CV AUC delta >= +0.003; revert otherwise. Check P@50 does not drop.

### sample_weight by label-trend magnitude (confidence weighting) (priority: high)

**How:** Inside each train fold: w = np.clip(np.abs(df_tr['trend_pct']), 5, 50)/50 (floor keeps near-flat rows at 0.1 weight, not zero); normalize; pass as logisticregression__sample_weight. Grid the floor in {0, 5, 10}. trend_pct is read ONLY to build train-fold weights, never enters X, never touches eval rows.

**Why:** Rows with trend_pct near the up/down boundary are effectively coin-flip labels; down-weighting them is label-noise cleaning, which is exactly what a 0.60-ceiling problem needs. This is the training-target analogue of the allowed 'label variants' idea and one of the strongest candidates in this dimension.

**Leakage:** trend_pct is label-window data, so the rule is strict: it may parameterize train-fold sample weights (the model already sees these rows' labels, which derive from the same column) but must NEVER appear as a feature, filter for eval rows, or anything computed at predict time. Harness must assert 'trend_pct' not in the feature list after the toggle.

**Accept:** Keep best floor if mean CV AUC delta >= +0.003 and P@50 non-degrading; revert otherwise.

### per-client rebalancing via sample_weight (priority: high)

**How:** Inside each train fold: n_c = df_tr.groupby('client_id')['client_id'].transform('size'); w = n_c.astype(float)**(-alpha) for alpha in {0.5, 1.0}; normalize w *= len(w)/w.sum(); fit with logisticregression__sample_weight=w. Extension point in the same grid: multiply by per-client class balance w *= 1/df_tr.groupby(['client_id', y_tr]).transform('size')**0.5 (equalize class within client too).

**Why:** 32 clients with almost certainly skewed sizes: unweighted training lets 2-3 giant clients dominate the coefficients, while grouped CV scores every held-out client's rows together. Equalizing client contribution directly targets the train/eval mismatch that grouped splitting creates. Best-motivated method in this dimension.

**Leakage:** None. client_id is used only to compute train-fold weights, never as a feature. Compute group sizes within the train fold, not on the full data.

**Accept:** Keep best alpha if mean CV AUC delta >= +0.003; also watch fold-level variance (this method should REDUCE the across-fold AUC spread even if the mean is flat; keep only on mean improvement per harness rule).

### per-client cap undersampling (priority: low)

**How:** Inside each train fold: cap = int(df_tr.groupby('client_id').size().median()) (grid: median, 2*median); for each client with more rows, keep a stratified-by-y random subsample of size cap via rng.choice per class; fit on the reduced train set. rng = np.random.default_rng(fold_seed).

**Why:** Hard version of per-client weighting. Strictly dominated by the weight version (throws away data) but included because weighting and subsampling occasionally differ for LR when big clients are also distributionally weird. Expect the weight version to win.

**Leakage:** None. Subsample train fold only; eval fold untouched and never subsampled.

**Accept:** Keep if mean CV AUC delta >= +0.003; skip entirely if per-client sample_weight already accepted (redundant).

### random undersampling of the POSITIVE majority to 50/50 (priority: low)

**How:** Inside each train fold: pos = np.where(y_tr==1)[0]; neg = np.where(y_tr==0)[0]; keep = np.concatenate([rng.choice(pos, size=len(neg), replace=False), neg]); fit pipeline on X_tr.iloc[keep]. Note the inversion: the class to shrink is the POSITIVE one.

**Why:** Classic imbalance tool applied to a barely-imbalanced problem (60/40) where the metric (AUC) is threshold-free: theory says roughly zero gain, and discarding ~20% of positives adds variance. Included so the harness can reject it cheaply.

**Leakage:** None. Resample train fold only; NEVER resample the eval fold (AUC on a resampled test set is meaningless).

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert.

### random oversampling of the NEGATIVE minority (priority: low)

**How:** Inside each train fold: neg = np.where(y_tr==0)[0]; extra = rng.choice(neg, size=len(pos)-len(neg), replace=True); fit on X_tr.iloc[np.concatenate([np.arange(len(y_tr)), extra])].

**Why:** For LogisticRegression this is mathematically ~equivalent to class_weight={0: pos/neg ratio}, so it cannot beat the class_weight grid and just costs memory. Included for completeness; expect revert.

**Leakage:** None. Train fold only.

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert. Skip if class_weight grid already evaluated (equivalent).

### SMOTE-style interpolation of negatives (NearestNeighbors + numpy) (priority: low)

**How:** Restructure the toggle to fit preprocessing first: Xt = pre.fit_transform(X_tr) (dense float array); Xn = Xt[y_tr==0]; nn = NearestNeighbors(n_neighbors=6).fit(Xn); _, idx = nn.kneighbors(Xn); n_new = (y_tr==1).sum() - (y_tr==0).sum(); pick anchors a = rng.integers(0, len(Xn), n_new), neighbors j = idx[a, rng.integers(1, 6, n_new)], lam = rng.random((n_new,1)); X_syn = Xn[a] + lam*(Xn[j]-Xn[a]); fit LogisticRegression on np.vstack([Xt, X_syn]) with y extended by zeros. Fractional one-hot values are fine for a linear model. NN on ~6k x ~30 dims runs in seconds with n_jobs=1.

**Why:** SMOTE helps when the minority class is tiny and the decision boundary is starved; at 40% minority with a linear model it adds essentially nothing beyond oversampling/class weights. Included to let the harness kill it with data.

**Leakage:** Synthesis strictly inside the train fold (both NN fit and sampling), so no eval rows leak in. Synthetic rows may interpolate across clients within the train fold — acceptable since eval clients are disjoint.

**Accept:** Keep if mean CV AUC delta >= +0.003 (stricter: extra moving parts); expected outcome: revert.

### Tomek-link cleaning of borderline positives (priority: low)

**How:** In preprocessed space: nn = NearestNeighbors(n_neighbors=1).fit(Xt); _, nb = nn.kneighbors(Xt) with self-exclusion (take 2 neighbors, use the second); tomek pairs = mutual nearest neighbors with opposite labels; drop the POSITIVE member of each pair; refit. One pass, ~seconds on 15k rows.

**Why:** Removes majority points sitting on the class boundary — a mild label-noise cleaner. On a problem whose whole difficulty IS boundary noise, this could nudge AUC, but the effect size for LR is usually tiny. Cheap to test.

**Leakage:** Train fold only; eval fold never cleaned.

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert or noise-level gain.

### Gaussian jitter augmentation of numerics (priority: low)

**How:** After preprocessing on train fold: stack k=2 noisy copies X_aug = np.vstack([Xt] + [Xt + rng.normal(0, s, Xt.shape) * (col_is_numeric_mask) for _ in range(k)]), y tiled; grid s in {0.05, 0.1} (features are standardized so s is in std units); one-hot columns get no noise. Fit LR on the stack.

**Why:** Noise augmentation on a linear model is equivalent to L2 regularization, which LR already has via C. Redundant with tuning C (a different dimension's job). Included honestly as a near-certain revert; only worth keeping in mind if the harness later adopts HistGradientBoostingClassifier, where jitter can act as a real regularizer.

**Leakage:** Train fold only; noise std computed from train-fold scaler.

**Accept:** Keep if mean CV AUC delta >= +0.003; expected outcome: revert.

### jittered oversampling of negatives (ROS + noise) (priority: low)

**How:** Combine the two above: duplicate negatives to 50/50, then add rng.normal(0, 0.05) to numeric columns of the DUPLICATED rows only. ~10 lines in the preprocessed space.

**Why:** The 'poor man's SMOTE'. Same verdict as SMOTE: minority is not starved, model is linear, expect nothing. Included so the enumeration is complete.

**Leakage:** Train fold only.

**Accept:** Keep if mean CV AUC delta >= +0.003; expected outcome: revert. Skip if both ROS and jitter already rejected.

### tabular mixup (same-label variant) (priority: low)

**How:** Sklearn LR rejects soft labels, so use label-preserving mixup: within each class of the train fold, sample pairs (i,j), lam ~ np.random.beta(0.4, 0.4), X_mix = lam*Xt[i] + (1-lam)*Xt[j], y_mix = y[i]; append n_mix = 0.5*len(Xt) rows. Cross-label alternative: mix any pair and draw y_mix ~ Bernoulli(lam) — noisier, include both in a 2-point grid.

**Why:** Mixup's gains come from regularizing flexible models; for LR, same-label mixup adds points inside the convex hull of each class, which barely moves a linear boundary. Honest expectation: no-op. Cheap to test.

**Leakage:** Pairs drawn within the train fold only; mixing across clients within the fold is fine (eval clients disjoint).

**Accept:** Keep if mean CV AUC delta >= +0.003; expected outcome: revert.

### bootstrap bagging of the pipeline (priority: low)

**How:** BaggingClassifier(estimator=clone(baseline_pipeline), n_estimators=20, max_samples=0.8, bootstrap=True, n_jobs=1, random_state=0). Score with predict_proba. Runtime ~20x baseline fit, still under a minute.

**Why:** Bagging reduces variance; LR on 15k rows is already low-variance, so gains are usually ~0.000-0.002 AUC. Slightly more interesting if the model dimension later swaps in a tree model. Included as the canonical bootstrap-resampling entry.

**Leakage:** None; bootstrap happens inside each CV train fold via the fitted bagger.

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert for LR.

### EasyEnsemble-style averaged undersampled models (priority: low)

**How:** for k in range(10): sample all negatives + len(neg) random positives from the train fold (rng seeded per k); fit clone(pipeline); collect predict_proba[:,1] on eval fold; final score = mean over k. Pure numpy loop, no imblearn.

**Why:** EasyEnsemble targets severe imbalance; at 60/40 the ensemble of undersampled LRs converges to roughly the class-weighted single model. Expect revert; costs 10 fits.

**Leakage:** Undersampling per train fold only.

**Accept:** Keep if mean CV AUC delta >= +0.003; expected outcome: revert.

### train-population widening: fit on all 30000 rows, evaluate on visible only (priority: high)

**How:** Load the full CSV without the impressions_90d>=100 filter. Honest CV recipe: (1) build folds ONCE with GroupKFold(5) over client_id on the FULL 30000 rows; (2) per fold, fit preprocessing + model on ALL train-fold rows (including sub-threshold); (3) predict on all test-fold rows but compute roc_auc_score ONLY on test rows where impressions_90d >= 100, and P@50 on the top-50 of those same visible test rows; (4) the baseline comparison must score the IDENTICAL visible test-row set — derive baseline folds from the same client-to-fold assignment, not a fresh split. impressions_90d is read only for the eval mask, never as a feature.

**Why:** +36% more training rows for free, and the sub-threshold pages still carry the same feature-label relationships. Two honest risks: their labels are noisier (a trend computed on <100 impressions is mostly count noise), and their feature distribution differs from the eval population, so plain widening can hurt — that is what the weighted variant below is for. Still one of the few methods that adds genuinely new information.

**Leakage:** Main trap is EVAL leakage via the mask: never filter or select eval rows by anything except the fixed impressions_90d>=100 population rule, and never let sub-threshold rows enter the AUC/P@50 computation. Selecting extra TRAIN rows by impressions_90d is population definition, not feature leakage, but note it conditions the train sample on a label-window quantity — acceptable because the eval set is conditioned the same way by construction.

**Accept:** Keep if mean CV AUC on the visible test subset improves >= +0.003 versus baseline computed on the identical visible test subset; revert otherwise.

### widening variant: down-weight the sub-threshold rows (priority: high)

**How:** Same CV recipe as plain widening, plus per-fold sample_weight: w = np.where(df_tr['impressions_90d'] >= 100, 1.0, beta) with beta grid {0.1, 0.25, 0.5}; normalize; pass to fit. Only toggled if run after (or instead of) plain widening.

**Why:** The realistic version of widening: extracts information from the extra 8k rows while discounting their noisy labels and off-population features. If plain widening is roughly neutral, this variant is the one likely to tip positive.

**Leakage:** Same as plain widening: impressions_90d builds train weights and the eval mask only, never a feature. Weights computed per train fold.

**Accept:** Keep best beta if mean visible-subset CV AUC delta >= +0.003 over the better of {baseline, plain widening}.

### widening variant: near-threshold band only (priority: medium)

**How:** Same recipe but add only rows with impressions_90d in [30, 100) (grid the lower edge in {10, 30, 50}) to the train folds, at full weight. Rows below the band are dropped from training entirely.

**Why:** Near-threshold pages are the least off-population and their labels are the least noisy among the sub-threshold set; deep-tail pages (impressions_90d ~ 0-10) have essentially random labels. A middle ground if full widening hurts but some widening helps.

**Leakage:** Same mask discipline as above; band selection applies to train folds only.

**Accept:** Keep best band if mean visible-subset CV AUC delta >= +0.003; skip if the down-weight variant already accepted (overlapping mechanism).

### 3-class training label (down/flat/up), score P(down) (priority: high)

**How:** Per train fold: y3 = df_tr['trend_direction'] (assumed 3 categories; if only 2 exist the toggle is a no-op — assert and skip). Fit the same pipeline with LogisticRegression (multinomial is the sklearn>=1.4 default). Score for ranking/eval: proba[:, list(clf.classes_).index('down')]. Eval unchanged: roc_auc_score(is_declining_test_visible, score).

**Why:** Collapsing flat+up into one negative class throws away structure: 'flat' pages plausibly sit between up and down in feature space, and letting the model separate them can sharpen the down-vs-rest boundary. Costs nothing, uses no extra columns as features. Genuine candidate.

**Leakage:** trend_direction is the label source, so using it as a richer TRAINING target on train folds is legitimate (same information tier as y itself). It must never be a feature and never touch eval rows except through the fixed binary label.

**Accept:** Keep if mean CV AUC delta >= +0.003 with P@50 non-degrading.

### regression surrogate target: rank by predicted -trend_pct (priority: medium)

**How:** Per train fold: t = np.clip(df_tr['trend_pct'], -100, 100); fit Ridge(alpha=1.0) or HistGradientBoostingRegressor(max_iter=200) on the same preprocessed features with target -t; ranking score = predictions; eval: roc_auc_score(binary label, score) on visible test rows. Grid: {Ridge, HGBRegressor} x clip at {50, 100}.

**Why:** The binary label discards magnitude; a page at -80% is a more informative training example than one at -2%. Regressing the continuous trend then thresholding implicitly at eval is a classic surrogate-target trick and sometimes beats classification on noisy binarized labels. Worth a real shot.

**Leakage:** trend_pct used ONLY as the train-fold target, same tier as the label itself; never a feature, never on eval rows. Harness must assert the feature matrix is identical to baseline's.

**Accept:** Keep best combo if mean CV AUC delta >= +0.003 (stricter: 4 grid points); tie-break on P@50.

### pseudo-labeling / self-training (SelfTrainingClassifier) (priority: low)

**How:** sklearn.semi_supervised.SelfTrainingClassifier(estimator=LogisticRegression(...), threshold=0.9) requires unlabeled rows marked y=-1. Only contrived construction available: mask the 7994 sub-threshold rows' labels to -1 and let self-training re-label them.

**Why:** Honest: there is NO genuinely unlabeled data — all 30000 rows have labels. Replacing real labels with model-generated pseudo-labels is strictly worse than train-population widening with the true labels. Enumerate-and-reject entry; the harness can skip it without running if widening was already tested.

**Leakage:** Would be train-fold-only, same discipline as widening.

**Accept:** Skip, or run once and revert unless mean CV AUC delta >= +0.003 (it will not be).

### recency sample_weight (priority: low)

**How:** Per train fold: w = np.exp(-df_tr['days_since_last_update'].fillna(df_tr['days_since_last_update'].median()) / h) for h in {180, 365}; normalize; pass as sample_weight.

**Why:** Explicitly requested variant. Honest: no strong story — there is no temporal train/test split here (split is by client), so 'recent pages are more like the eval distribution' does not apply. Weak candidate, cheap to test.

**Leakage:** None; days_since_last_update is an allowed feature already.

**Accept:** Keep best h if mean CV AUC delta >= +0.003; expected outcome: revert.

### down-weight rows with many imputed fields (priority: low)

**How:** Per train fold: n_miss = X_tr[allowed_numeric_cols].isna().sum(axis=1); w = 1.0/(1.0 + 0.25*n_miss); normalize; pass as sample_weight.

**Why:** Rows where word_count/search_volume/etc. are fillna(0) contribute distorted feature values; down-weighting them reduces imputation damage. Marginal mechanism since missing indicators already exist as features; expect near-zero delta.

**Leakage:** None; missingness computed per train fold from allowed columns.

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert.

### test-time augmentation (average over jittered copies) (priority: low)

**How:** At predict time only: p = np.mean([pipe.predict_proba(jitter(X_te))[:,1] for _ in range(8)], axis=0) where jitter adds rng.normal(0, 0.05*col_std) to raw numeric columns before the pipeline (compute col_std from train fold). No refitting.

**Why:** Honest: for LogisticRegression this is a provable near-no-op — averaging sigmoids of symmetrically jittered linear scores preserves ranking almost exactly, so AUC will not move. Only becomes non-trivial if the model dimension adopts HistGradientBoosting, where TTA smooths step-function predictions and can add ~0.001-0.003 AUC. Runs in seconds.

**Leakage:** None, but jitter scale must come from train-fold statistics, and TTA must apply identically to all eval rows (never selectively).

**Accept:** Keep if mean CV AUC delta >= +0.002; expected outcome: revert under LR, retry if the kept model becomes tree-based.

