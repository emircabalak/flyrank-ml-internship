### LogisticRegression l2 C-tuning (baseline retune) (priority: high)

**How:** Keep baseline pipeline (fillna0 + indicators + log1p + one-hot + StandardScaler). Coarse: LogisticRegression(solver='lbfgs', max_iter=2000, C in np.logspace(-3,2,6), class_weight in [None,'balanced']). Fine: 5 points at factor-2 spacing around best C. 12 fits coarse, <5s each at 15k x ~30.

**Why:** Baseline C=1.0 may be off; class_weight matters little for AUC (monotone shift) but can move P@50. Cheapest possible win; often a no-op but must be checked first so later toggles compare against a tuned linear reference.

**Leakage:** none (tune inside grouped CV folds only)

**Accept:** Keep best (C, class_weight) if mean grouped-CV AUC delta > +0.002 vs baseline and non-negative in majority of folds; else revert to C=1, balanced.

### LogisticRegression l1 / elasticnet via saga (priority: medium)

**How:** Same pipeline, LogisticRegression(solver='saga', max_iter=5000, tol=1e-3). Coarse: penalty in ['l1','elasticnet'], l1_ratio in [0.2,0.5,0.8] (elasticnet only), C in np.logspace(-3,1,5), class_weight='balanced'. Fine: C around best. Scaled input already present (saga needs it).

**Why:** Sparse penalty can zero noisy one-hot levels (e.g. rare model_used categories). With only ~30 features, effect is likely tiny; include because it is cheap and occasionally cleans up weak dummies.

**Leakage:** none

**Accept:** Keep if mean grouped-CV AUC delta > +0.002; else revert to l2.

### RidgeClassifier (priority: low)

**How:** Swap final estimator: RidgeClassifier(alpha in np.logspace(-2,3,6), class_weight='balanced'). Rank with decision_function (no predict_proba; AUC fine, P@50 = top-50 by decision_function).

**Why:** Squared-loss linear classifier; on this data it should track logistic within noise. Expected no-op, but costs seconds to reject.

**Leakage:** none

**Accept:** Keep only if mean grouped-CV AUC delta > +0.002 (unlikely); revert otherwise.

### SGDClassifier (log_loss / modified_huber) (priority: low)

**How:** SGDClassifier(loss in ['log_loss','modified_huber'], alpha in np.logspace(-6,-2,5), penalty in ['l2','elasticnet'], max_iter=2000, tol=1e-4, early_stopping=False, class_weight='balanced', random_state=0). Scaled input mandatory (already in pipeline). Rank by predict_proba (log_loss) or decision_function.

**Why:** Same hypothesis class as tuned LogReg, different optimizer; only wins via modified_huber's outlier robustness, which is improbable here. Cheap reject.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### LinearDiscriminantAnalysis (svd, lsqr+shrinkage) (priority: medium)

**How:** Coarse: LDA(solver='svd') and LDA(solver='lsqr', shrinkage='auto'). Fine: shrinkage in [0.05,0.1,0.3,0.5]. Use predict_proba. One-hots violate Gaussian assumption but LDA is a linear scorer and tolerates it; keep scaler.

**Why:** Shrinkage covariance can regularize better than l2 on correlated numerics (word_count/char_count, prev-30d counts are collinear). Small chance of a marginal win; near-free to test.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002.

### QuadraticDiscriminantAnalysis (priority: low)

**How:** QDA(reg_param in [0.1,0.3,0.5,0.7,0.9]). reg_param>0 mandatory: one-hot columns make per-class covariance singular (reg_param=0 will warn/collapse). Consider running on numerics-only view (drop one-hots) as a second toggle variant.

**Why:** Captures class-dependent covariance; with 60/40 classes and weak signal it usually overfits or degenerates on dummies. Include to reject cheaply.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### GaussianNB (priority: low)

**How:** GaussianNB(var_smoothing in np.logspace(-12,-3,10)) on the scaled matrix. predict_proba native.

**Why:** Feature-independence assumption is badly wrong (collinear counts, one-hots); NB probabilities are miscalibrated but ranking sometimes survives. Expected below 0.60; seconds to reject.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### BernoulliNB / ComplementNB on discretized view (priority: low)

**How:** Separate branch: ColumnTransformer with existing one-hots + KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile') on the 10 numerics (fillna handled before), then BernoulliNB(alpha in [0.1,0.5,1.0]) or ComplementNB(alpha same; needs non-negative input, satisfied by one-hot). 

**Why:** Discretization gives NB a fairer shot and adds mild nonlinearity in numerics. Still expected to trail linear; included for exhaustiveness.

**Leakage:** none (KBinsDiscretizer fit inside pipeline per fold)

**Accept:** Keep if AUC delta > +0.002; expected revert.

### KNeighborsClassifier (priority: low)

**How:** Scaled input mandatory (baseline scaler stays). Coarse: n_neighbors in [50,100,200,400], weights in ['uniform','distance'], algorithm='brute', metric='euclidean', n_jobs=1. Fine: k around best. Rank by predict_proba[:,1]; large k needed for smooth ranking at 22k rows. Cost per fold ~15k x 6.6k x 30 brute-force distances, tens of seconds — acceptable but the slowest of the cheap models.

**Why:** Local structure in (search_volume, cpc, prev-30d) space could carry signal a linear model misses, but euclidean distance over mixed one-hot/numeric space is usually mush. Low expectation.

**Leakage:** none. Note: same-client pages are near-duplicates in feature space, so kNN partly memorizes client identity via feature clusters — grouped CV makes this honest (neighbors from train clients only), so any AUC gain is real cross-client signal, not leakage.

**Accept:** Keep if AUC delta > +0.002.

### SVC rbf (subsample or Nystroem approximation) (priority: low)

**How:** Full 15k SVC grid is impractical (O(n^2)+ kernel, probability=True adds internal 5-fold — never enable it; rank by decision_function). Route A (subsample): per fold, train on 4000 stratified-sampled rows (random_state=0), SVC(kernel='rbf', C in [0.1,1,10], gamma in ['scale',0.01,0.1], class_weight='balanced'), score full validation with decision_function. Route B (preferred, full data): Nystroem(kernel='rbf', gamma in [0.01,0.033,0.1], n_components=300, random_state=0) + LogisticRegression(C in [0.1,1,10]) — linear cost, approximates rbf-SVM ranking.

**Why:** Kernel nonlinearity is the honest way to test whether the ceiling above 0.60 is nonlinear structure. HGB usually dominates it on tabular data, so this is a secondary check; Route B is the only variant worth the harness's time.

**Leakage:** none (subsampling and Nystroem fit within each training fold)

**Accept:** Keep if AUC delta > +0.002; if Route A wins but Route B doesn't, distrust it (subsample noise) and require the gain on 2 different subsample seeds.

### LinearSVC (priority: low)

**How:** LinearSVC(C in np.logspace(-3,1,5), class_weight='balanced', dual=False, max_iter=5000) on scaled matrix. Rank by decision_function directly — CalibratedClassifierCV is NOT needed for AUC or P@50 (both rank-based; sigmoid calibration is monotone and changes neither). Only wrap in calibration if this model later enters a probability-averaged blend.

**Why:** Hinge loss vs log loss on the same linear class; occasionally differs by a hair. Expected ≈ tuned LogReg.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### NuSVC (priority: low)

**How:** Reject on cost: same kernel complexity as SVC with extra nu-feasibility failures under class imbalance (nu must be <= 2*min(class ratio)... fits error out for many nu values). If the harness insists: 3000-row subsample, nu in [0.3,0.5], gamma='scale', rank by decision_function.

**Why:** Strictly dominated by the SVC entry — same model family, worse parametrization for imbalanced data. Listed only for completeness; recommend skipping entirely.

**Leakage:** none

**Accept:** Skip; if run, same +0.002 rule.

### DecisionTreeClassifier (priority: low)

**How:** DecisionTreeClassifier(max_depth in [3,4,5,6,8], min_samples_leaf in [50,100,200], class_weight='balanced', random_state=0). No scaler needed but harmless to keep the pipeline. predict_proba is piecewise-constant — many ties, AUC still valid.

**Why:** Will not beat ensembles; useful as a sanity floor for how much axis-aligned nonlinearity exists. Near-certain revert.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### RandomForestClassifier (priority: high)

**How:** Coarse: n_estimators=300, max_depth in [None,8,16], min_samples_leaf in [5,20,50], max_features in ['sqrt',0.5], class_weight='balanced_subsample', n_jobs=1, random_state=0 (12 combos, ~30-60s each). Fine: n_estimators=500 around best, min_samples_leaf +/- one step. Rank by predict_proba[:,1]. Feed the un-scaled branch (scaling irrelevant for trees); keep fillna+indicators since RF has no native NaN handling.

**Why:** Standard strong tabular baseline; interactions like content_age_days x prev-30d trend are exactly what it finds. Main honest competitor to HGB.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002 and per-fold improvements are not driven by a single client-heavy fold.

### ExtraTreesClassifier (priority: medium)

**How:** Same grid as RF: n_estimators=300 (500 fine), max_depth in [None,8,16], min_samples_leaf in [5,20,50], max_features in ['sqrt',0.5], class_weight='balanced_subsample', n_jobs=1, random_state=0. Faster per tree than RF (random splits).

**Why:** Extra randomization helps when features are noisy proxies, which describes this pool. Sometimes edges out RF on weak-signal data.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002.

### GradientBoostingClassifier (priority: low)

**How:** GradientBoostingClassifier(learning_rate in [0.03,0.1], n_estimators in [200,500], max_depth in [2,3], subsample in [0.7,1.0], min_samples_leaf=50, random_state=0). ~1-3 min per fit at the large end; trim grid if slow.

**Why:** Almost strictly dominated by HistGradientBoosting (same algorithm family, slower, no NaN/categorical support). Include only so the harness has proof; run AFTER HGB and skip if HGB already accepted.

**Leakage:** none

**Accept:** Keep only if it beats both baseline and HGB by > +0.002 (very unlikely).

### HistGradientBoostingClassifier (top candidate) (priority: high)

**How:** Own preprocessing branch exploiting native NaN + categoricals: numerics passed RAW with NaN (drop fillna(0) and log1p — trees don't need them; keep has_* indicators optional as a sub-toggle since HGB's NaN branch may subsume them), categoricals via OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).astype(float) with categorical_features=[bool mask], or with sklearn>=1.4 cast to pandas 'category' dtype and set categorical_features='from_dtype'. No scaler. Coarse grid: learning_rate in [0.05,0.1], max_leaf_nodes in [15,31,63], l2_regularization in [0.0,1.0,10.0], min_samples_leaf in [20,50], max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, scoring='roc_auc', class_weight='balanced', random_state=0 (36 combos, ~10-30s each at n_jobs=1). Fine: max_depth in [None,4,8] and max_bins in [128,255] around best. Purity variant: early_stopping=False with max_iter fixed at the coarse-stage median stopping point, because the internal validation split is random (not client-grouped).

**Why:** Best available booster given no xgboost/lightgbm; handles the heavy missingness (word_count 7699 nulls, provider_used 21438 nulls) natively instead of via fillna(0) distortion, and learns splits like 'word_count missing' as signal. Most likely single-model AUC gain in the whole enumeration.

**Leakage:** Internal early-stopping validation_fraction split ignores client groups — same-client rows in its train/val can make it stop slightly late/early, but no label information crosses into the OUTER holdout, so measured CV AUC stays honest. The early_stopping=False variant removes even that wrinkle.

**Accept:** Keep if mean grouped-CV AUC delta > +0.002 and P@50 does not drop > 0.02.

### AdaBoostClassifier (priority: low)

**How:** AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth in [1,2]), n_estimators in [200,500], learning_rate in [0.1,0.5,1.0], algorithm='SAMME', random_state=0). sklearn>=1.4: SAMME.R is deprecated/removed — use SAMME. predict_proba is squashed toward 0.5 but rank-valid.

**Why:** Exponential loss on weak-signal noisy labels tends to chase mislabeled points; historically loses to gradient boosting. Cheap reject after HGB.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002 vs current best; expected revert.

### BaggingClassifier over LogisticRegression (priority: low)

**How:** BaggingClassifier(estimator=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(C=best,class_weight='balanced',max_iter=2000))]), n_estimators=25, max_samples=0.8, max_features in [0.7,1.0], bootstrap=True, n_jobs=1, random_state=0). Average predict_proba is automatic.

**Why:** LogReg is a low-variance estimator, so bagging over rows is a near-certain no-op; max_features<1 adds a random-subspace flavor that very occasionally helps. Cheap to reject.

**Leakage:** none

**Accept:** Keep if AUC delta > +0.002; expected revert.

### MLPClassifier (priority: low)

**How:** Scaled branch mandatory. MLPClassifier(hidden_layer_sizes in [(32,),(64,),(64,32)], alpha in [1e-4,1e-3,1e-2], learning_rate_init=1e-3, batch_size=256, max_iter=300, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=0). ~30-90s per fit. Fine: widen best architecture, tune learning_rate_init in [3e-4,3e-3].

**Why:** On 15k rows of weak-signal tabular data, MLPs typically match logreg at best and are seed-noisy. Worth one coarse pass because it is the only smooth nonlinear model besides kernels.

**Leakage:** Same non-grouped internal early-stopping split caveat as HGB — affects stopping only, outer CV remains honest.

**Accept:** Keep if AUC delta > +0.002 averaged over 3 seeds (single-seed MLP deltas are noise).

### GaussianProcessClassifier (priority: low)

**How:** REJECT on cost: O(n^3) Laplace approximation is hopeless at 15k rows (would need hours+ and gigabytes). Desperation variant only: 1500-row stratified subsample per fold, GPC(kernel=1.0*RBF(1.0), n_jobs=1) — and at that size it just approximates the SVC-subsample entry with worse scaling.

**Why:** Listed for completeness; the Nystroem+LogReg entry already tests the same rbf hypothesis at tractable cost. Recommend the harness skips this.

**Leakage:** none

**Accept:** Skip.

### Soft VotingClassifier over top models (priority: high)

**How:** After single-model toggles settle, VotingClassifier(estimators=[('lr', best_logreg_pipe), ('hgb', best_hgb_pipe), ('rf', best_rf_pipe)], voting='soft', weights in [None,(1,2,1),(1,2,2)], n_jobs=1). Each base keeps its own preprocessing pipeline inside the tuple. Only include bases with predict_proba. Coarse: top-2 models; fine: top-3 and weight variants.

**Why:** Linear + boosted models make decorrelated errors; averaging their probabilities is the classic cheap +0.002-0.01 on weak-signal problems. One of the likeliest accepted toggles after HGB.

**Leakage:** none (pure refit-and-average, no internal CV)

**Accept:** Keep if mean grouped-CV AUC beats the best single member by > +0.002.

### Rank-average blending of top-k models (priority: high)

**How:** Per CV fold: fit top-k (k=2,3) models, get validation scores s_m, convert to ranks r_m = pd.Series(s_m).rank(method='average'), blend = mean(r_m across models), compute AUC/P@50 on blend. Pure numpy/pandas, no estimator wrapper needed — implement as a harness-level scorer over stored per-model fold predictions so no refits are required.

**Why:** Rank transform removes probability-scale mismatch (HGB vs uncalibrated LinearSVC decision_function etc.), so models rankable only by decision_function can join the blend without calibration. Usually within noise of soft voting but strictly more robust; nearly free if per-fold predictions are cached.

**Leakage:** none — blending uses only same-fold validation predictions; no fitting on validation data (equal weights are fixed a priori).

**Accept:** Keep if blend mean CV AUC beats best single model by > +0.002.

### Weighted-score blending with fold-tuned weights (priority: medium)

**How:** On rank-normalized per-fold predictions (from cache): for 2 models sweep w in np.arange(0,1.05,0.05), blend = w*r1+(1-w)*r2; for 3 models sweep a coarse simplex grid (step 0.1, ~66 points). CRITICAL: choose w by leave-one-fold-out — pick w maximizing mean AUC over k-1 folds, evaluate on the held-out fold, average those held-out AUCs. Never pick w on the same folds you report.

**Why:** Squeezes the last drop beyond equal-weight blending; on weak signal the tuned w is noisy and often collapses to ~equal weights, in which case revert to the simpler rank-average.

**Leakage:** Weight-tuning on the reporting folds is selection leakage (optimistic AUC) — the leave-one-fold-out protocol above avoids it. No label leakage.

**Accept:** Keep only if leave-one-fold-out AUC beats equal-weight rank-average by > +0.002; otherwise prefer the simpler blend.

### Stacking (manual group-aware OOF; StackingClassifier caveat) (priority: medium)

**How:** sklearn LIMITATION: StackingClassifier(cv=5) uses (Stratified)KFold internally and its fit() does not accept groups — even passing cv=GroupKFold(5) fails at split time because no groups reach it (metadata routing for this is experimental/version-fragile). Do it manually per outer fold: gkf=GroupKFold(5); oof=np.zeros((len(Xtr),M)); for tr,va in gkf.split(Xtr,ytr,groups=grp_tr): for m,pipe in enumerate(bases): oof[va,m]=clone(pipe).fit(Xtr.iloc[tr],ytr.iloc[tr]).predict_proba(Xtr.iloc[va])[:,1]; meta=LogisticRegression(C=1.0).fit(oof,ytr); then refit each base on the full outer-train, build test matrix of base predict_proba on outer-validation, score meta.predict_proba(test_matrix)[:,1]. Bases: best 2-3 models. Cost: M*(5+1) fits per outer fold — budget accordingly, coarse hyperparams frozen from earlier toggles.

**Why:** A learned meta-combination can beat fixed-weight blending when base models' relative strength varies by region of feature space. With ~0.60-AUC bases the meta signal is thin; expect it to roughly tie rank-averaging, but it is the principled version and worth one run.

**Leakage:** Using stock StackingClassifier would train the meta-learner on OOF predictions from non-grouped folds — same-client rows in inner train/val inflate base OOF quality and mistrain the meta weights (optimism, not outer-holdout leakage). The manual GroupKFold OOF recipe above eliminates this; only run the manual version.

**Accept:** Keep if mean outer grouped-CV AUC beats best blend by > +0.002; given cost, also require it in >= half the folds.

### CalibratedClassifierCV (sigmoid / isotonic) (priority: low)

**How:** CalibratedClassifierCV(estimator=best_pipe, method in ['sigmoid','isotonic'], cv=5, n_jobs=1). NOTE for single models: sigmoid (Platt) is strictly monotone, so ROC-AUC and P@50 are mathematically UNCHANGED — as a standalone toggle it can only be a no-op; isotonic creates ties and can shift AUC by ~+/-0.001 noise. Its real role: mapping heterogeneous scores to comparable probabilities BEFORE soft-voting/weighted probability blending (rank-average blending makes even that unnecessary). Also: internal cv is not group-aware and fit() takes no groups — for a group-honest version calibrate manually on GroupKFold OOF scores with IsotonicRegression(out_of_bounds='clip') or sigmoid via LogisticRegression on the score.

**Why:** Include so the harness documents the no-op rather than wondering; only test it as a preprocessing step for the probability-blend entries, never expecting standalone AUC gain.

**Leakage:** Non-grouped internal CV gives optimistic calibration curves (same-client rows), harmless for AUC but distorts blend weights slightly; use the manual grouped recipe if calibrated blending is kept.

**Accept:** Standalone: auto-revert (AUC-invariant). Within blending: keep only if calibrated soft-vote beats rank-average blend by > +0.002.

### Multi-seed averaging of one stochastic model (priority: medium)

**How:** For the best stochastic model (HGB, RF, ExtraTrees, or MLP): fit with random_state in [0,1,2,3,4], average predict_proba[:,1] across seeds per fold, score the average. Pure loop + np.mean(axis=0); 5x fit cost of one model. Deterministic models (LogReg lbfgs, Ridge, LDA) are exact no-ops — skip them.

**Why:** Removes seed variance from subsampled-bin/bootstrap randomness; typically +0.000-0.003 AUC, and it also stabilizes the harness's own accept/revert decisions on later toggles. Cheap and boring.

**Leakage:** none

**Accept:** Keep if mean grouped-CV AUC delta > +0.001 (lower bar: it cannot overfit, only denoise) and runtime stays acceptable.

