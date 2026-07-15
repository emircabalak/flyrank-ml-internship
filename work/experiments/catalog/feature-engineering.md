### prev_window_ratio_features (priority: high)

**How:** From the 3 allowed prev-30d counts build: prev_ctr = np.where(imp>0, clicks/imp, 0); sessions_per_impression = np.where(imp>0, sess/imp, 0); clicks_per_session = np.where(sess>0, clicks/sess, 0), with imp=impressions_prev_30d etc. Clip each to [0, p99 of train] to tame division blowups. Implemented as a plain DataFrame step before the ColumnTransformer; scaled by the existing StandardScaler.

**Why:** Raw counts mostly encode page size/site size; ratios encode page quality independent of scale, which is exactly what transfers across clients under grouped CV. prev_ctr is the single most SEO-meaningful allowed derivation. Best single bet in this dimension.

**Leakage:** None: uses only prev-window columns explicitly declared leakage-safe. Ratios are row-local, no fitting.

**Accept:** Toggle on baseline; KEEP if mean GroupKFold(5, groups=client_id) AUC delta >= +0.002 and P@50 drop <= 0.02, else REVERT.

### zero_prev_traffic_flags (priority: high)

**How:** Binary columns: (impressions_prev_30d==0).astype(int), same for clicks_prev_30d, sessions_prev_30d, plus all_prev_zero = product of the three. Row-local numpy, 4 columns.

**Why:** log1p(0)=0 conflates 'zero' with 'tiny'; pages with literally no prior-window traffic are a qualitatively different regime (new/dead pages) that a linear model can't isolate from the continuous feature. Cheap, low-risk.

**Leakage:** None; prev-window only, row-local.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### staleness_and_update_recency (priority: high)

**How:** staleness_ratio = np.clip(days_since_last_update / np.maximum(content_age_days,1), 0, 1); update_recency_days = content_age_days - days_since_last_update (time between publish and last update; a pure difference feature). Two columns, row-local.

**Why:** Decline is the label; content staleness is the canonical causal story for SEO decline. The ratio normalizes 'never updated' (ratio~1) vs 'recently refreshed' (ratio~0) independent of page age, which raw days columns can't express linearly.

**Leakage:** None; static page properties only.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### content_density_ratios (priority: medium)

**How:** avg_word_len = np.where(word_count>0, char_count/word_count, 0); words_per_day_of_age = word_count/(content_age_days+1); imp_prev_per_100_words = impressions_prev_30d/((word_count/100)+1). Where word_count is null (7699 rows) set ratios to 0 and rely on the existing has_word_count indicator.

**Why:** Traffic-per-word separates thin pages coasting on volume from substantial pages; avg word length is a weak content-style proxy. Plausible small gains; imp_per_100_words is the strongest of the three.

**Leakage:** None; allowed columns only, row-local.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### keyword_opportunity_composites (priority: high)

**How:** sv_x_cpc = log1p(search_volume)*cpc (commercial value of the keyword); competition_x_cpc = competition*cpc; competition_x_log_sv = competition*log1p(search_volume); staleness_x_log_sv = days_since_last_update*log1p(search_volume) (high-value page going stale). Nulls -> 0 with existing has_keyword indicator. 4 columns.

**Why:** Hand-built domain interactions: pages targeting valuable, competitive keywords decay faster when neglected because competitors refresh. These are the specific cross-terms a linear model needs spelled out.

**Leakage:** None; static columns, row-local.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### log1p_all_skewed_numerics (priority: high)

**How:** Extend the baseline's log1p (currently only 3 prev counts) to search_volume, cpc, word_count, char_count, content_age_days, days_since_last_update via np.log1p after fillna(0). Replace raw columns, don't duplicate.

**Why:** search_volume and the count columns are heavy-tailed; StandardScaler on raw values lets a few huge rows dominate the LR gradient. Standard fix, near-free.

**Leakage:** None; deterministic row-local transform.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### yeo_johnson_or_sqrt_alternative (priority: low)

**How:** sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True) on all numerics inside the pipeline (fit on train folds only); box-cox not usable (zeros). Alternative arm: np.sqrt on the count columns instead of log1p.

**Why:** Largely redundant with log1p; included so the harness can cheaply confirm log1p is the right member of the family. Expect no gain over the log1p method; test only if log1p was kept.

**Leakage:** None if fit inside the CV pipeline (PowerTransformer learns lambda from train fold only).

**Accept:** KEEP only if it beats the log1p variant by >= +0.002 mean grouped-CV AUC, else REVERT.

### quantile_transform_global (priority: medium)

**How:** sklearn.preprocessing.QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, n_train), subsample=None, random_state=0) on all numerics, inside the Pipeline so it fits on train folds only.

**Why:** Rank-normalizes tails and outliers wholesale. Caveat under grouped split: quantiles are fit on train clients, and client-level distribution shift means test-client values map through miscalibrated quantiles; the per-client rank method below is shift-invariant and should beat this. Included for cheap comparison.

**Leakage:** None when fit inside the pipeline. Fitting on the full dataset before splitting would leak test-client distributions; do not do that.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT; if both this and per_client_rank win, keep only the better one.

### per_client_rank_features (priority: high)

**How:** df.groupby('client_id')[cols].rank(pct=True) for cols = [impressions_prev_30d, clicks_prev_30d, sessions_prev_30d, search_volume, word_count, days_since_last_update, content_age_days]; append as new columns (keep originals). Computed statelessly on the raw frame BEFORE the split: each row's value depends only on rows of its own client, and grouped CV keeps clients whole, so train/test are unaffected by each other.

**Why:** Normalizes away cross-client scale differences (big site vs small site), which is the main obstacle to grouped generalization at a 0.60 ceiling. 'Is this page in the bottom quartile of its own site's traffic' transfers across clients where raw counts don't. Deployment-honest: scoring is per-site batch, so within-client ranks are computable at inference.

**Leakage:** None: label-free and client-local. The only subtlety is the population filter (impressions_90d>=100) selecting which rows exist per client; that filter defines the population identically at train and inference, per the stated rules.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### client_aggregate_features (priority: high)

**How:** Stateless groupby('client_id') transforms on the raw frame: n_pages = transform('size'); client_mean_log_imp_prev = transform of log1p(impressions_prev_30d).mean(); page_traffic_share = impressions_prev_30d / (client sum impressions_prev_30d + 1); dev_word_count = word_count - client mean word_count; dev_log_imp = log1p(imp_prev) - client mean log1p(imp_prev). LEGIT because label-free and computable at inference from the client's own scoring batch. LEAKY and forbidden: any client aggregate of the label or forbidden columns (client decline rate, client mean trend_pct/ctr/avg_position), target-encoded client_id, and client_id one-hot (not leaky but useless: unseen test clients get all-zeros).

**Why:** Client context matters (a page underperforming ITS site differs from a small site's normal page), and deviation features encode categorical-by-numeric interaction with client as the group. n_pages and traffic_share are the strongest candidates. Some redundancy with per_client_rank; harness will arbitrate.

**Leakage:** None for the listed aggregates: label-free, client-local, clients never straddle folds. Anything aggregating label/forbidden columns is leakage by construction; excluded.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### hand_picked_pairwise_products (priority: medium)

**How:** ~8 products on already-transformed features: log_imp_prev*staleness_ratio, prev_ctr*log1p(search_volume), prev_ctr*competition, log1p(word_count)*log1p(content_age_days), log_imp_prev*log1p(search_volume), has_provider*days_since_last_update, log_sessions_prev*staleness_ratio, competition*staleness_ratio. Plain numpy column products before scaling.

**Why:** Targeted second-order terms LR can't learn itself; picked to encode decline stories (busy page going stale, high-CTR page in a competitive niche). Cheaper and less overfit-prone than full PolynomialFeatures.

**Leakage:** None; products of allowed row-local features.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### polynomial_interactions_full (priority: medium)

**How:** sklearn.preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False) applied to the ~13 log-transformed numerics inside the numeric branch (before StandardScaler): 13 -> ~91 columns. Runs in seconds on 22k rows.

**Why:** Exhaustive version of the previous method; with n=22k and LR it may squeeze a bit more, but grouped CV punishes interactions that only hold within train clients, so expect small or negative delta. Kept as the brute-force comparator to hand-picked products.

**Leakage:** None; deterministic expansion of allowed features.

**Accept:** Higher bar for a 7x feature blow-up: KEEP only if mean grouped-CV AUC delta >= +0.003, else REVERT; skip if hand_picked_pairwise_products already kept and this adds < +0.001 over it.

### squared_terms_only (priority: low)

**How:** Append np.square of the log-transformed continuous numerics (log_imp_prev, log_sv, log_word_count, log1p(content_age_days), log1p(days_since_last_update), cpc, competition): 7 extra columns.

**Why:** Captures simple U-shapes (mid-age content declining most). Splines below subsume this; expect near-zero delta. Included because it is nearly free to test.

**Leakage:** None; row-local.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT; drop if splines kept.

### spline_basis_expansion (priority: high)

**How:** sklearn.preprocessing.SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation='constant') on the continuous log-transformed numerics only (not binary flags), inside the numeric ColumnTransformer branch, followed by StandardScaler. ~13 numerics -> ~91 columns.

**Why:** LR + splines = GAM: the single most reliable nonlinearity upgrade for a linear model when moving to trees is a separate dimension. Monotone-but-saturating effects (traffic, age) are likely here. Best medium-effort bet after ratios.

**Leakage:** None; knots are fit on train folds inside the pipeline.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.003 (feature-count penalty), else REVERT.

### kbins_discretization (priority: medium)

**How:** sklearn.preprocessing.KBinsDiscretizer(n_bins=8, encode='onehot-dense', strategy='quantile', subsample=None) on the continuous numerics as an ADDED branch (keep the scaled originals). Optional second arm: strategy='kmeans', n_bins=6.

**Why:** Piecewise-constant nonlinearity; usually dominated by splines but occasionally wins on threshold-like effects (e.g. 'updated within 30 days'). Cheap toggle.

**Leakage:** None; bin edges fit on train folds inside the pipeline.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT; drop if splines kept and this adds nothing over them.

### missing_indicator_expansion (priority: high)

**How:** Beyond baseline has_keyword/has_word_count add: has_provider = provider_used.notnull() (only 8562 non-null of 30000 — presence means 'page was AI-refreshed', likely a real treatment signal); has_model = model_used.notnull(); has_intent = main_intent.notnull(); has_competition_level = competition_level.notnull(); n_missing = row-wise count of nulls across all null-bearing allowed columns. 5 columns of .isnull().astype(int).

**Why:** provider_used's 21438 nulls are almost certainly informative missingness (whether a refresh tool touched the page), directly related to decline. n_missing proxies data-coverage/page-importance. Highest value-per-line-of-code method here.

**Leakage:** None: missingness of allowed static columns. Verify provider_used missingness is not mechanically derived from the label window; it is a static ops field, so no.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### median_impute_instead_of_zero (priority: high)

**How:** sklearn.impute.SimpleImputer(strategy='median', add_indicator=False) for search_volume, competition, cpc, word_count, char_count inside the pipeline (indicators already exist), replacing fillna(0). Keep fillna(0) for prev counts where 0 is meaningful.

**Why:** fillna(0) injects a fake extreme value for search_volume/cpc (null means 'no keyword data', not zero volume), distorting the linear fit; median keeps the imputed rows neutral while the indicator carries the missingness signal.

**Leakage:** None; medians fit on train folds inside the pipeline.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### one_hot_provider_and_model_used (priority: high)

**How:** Add provider_used (2 cats) and model_used (5 cats) to the existing OneHotEncoder block with a filled 'missing' category: df[col].fillna('missing'), OneHotEncoder(handle_unknown='ignore'). ~9 extra columns.

**Why:** These categoricals are absent from the baseline entirely; which model/provider refreshed a page could correlate with refresh quality and thus decline. Overlaps with has_provider/has_model indicators; harness picks the winner.

**Leakage:** None; static allowed categoricals.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### one_hot_tier_categoricals (priority: medium)

**How:** Add freshness_tier, age_tier, word_count_tier, char_count_tier (4 cats each, fillna('missing')) to the OneHotEncoder block. ~17 extra columns.

**Why:** Tiers are precomputed binnings of numerics already in the model, so mostly redundant; but they may encode thresholds chosen with domain knowledge, and word/char tiers are partially available where raw counts are null. Modest expectations.

**Leakage:** None; static allowed categoricals derived from allowed numerics.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT; mutually arbitrate with tier_ordinal_encoding.

### tier_ordinal_encoding (priority: medium)

**How:** Map ordered tiers to ints instead of (or alongside) one-hot: sklearn.preprocessing.OrdinalEncoder(categories=[explicit ordered lists], handle_unknown='use_encoded_value', unknown_value=-1) for freshness_tier, word_count_tier, char_count_tier, competition_level; age_tier already has age_tier_order — just add that raw column if not present. Then StandardScaler.

**Why:** Ordinal tiers are monotone by construction; a single scaled ordinal costs 1 dof vs 4 for one-hot, helping a regularized LR. Small but real chance of a gain, especially competition_level as ordinal.

**Leakage:** None; deterministic mapping.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### frequency_encoding_categoricals (priority: low)

**How:** Custom transformer: fit stores value_counts(normalize=True) per categorical (content_type, main_intent, competition_level, model_used) on the train fold; transform maps categories to frequencies, unseen -> 0. Append as numeric columns.

**Why:** With 3-5 levels per categorical, one-hot already spans everything frequency encoding can express; this only helps at high cardinality. Expect zero delta; included for completeness since it is 10 lines.

**Leakage:** None; frequencies are label-free and fit on train folds.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### target_encoding_group_oof (priority: low)

**How:** Manual out-of-fold scheme (sklearn TargetEncoder's internal KFold is NOT group-aware, so same-client rows inform each other's encodings; do not use it). Per outer training fold: inner = GroupKFold(5).split(X_tr, y_tr, groups=client_tr); for each inner split compute smoothed category means on inner-train: enc[c]=(sum_y[c]+m*global_mean)/(n[c]+m), m=20; assign encodings to inner-held-out rows so no row is encoded using its own client's labels. Validation/test rows get full-training-fold encodings. Apply to content_type, main_intent, competition_level, model_used.

**Why:** Honest expectation: with low-cardinality categoricals and LR, target encoding is information-equivalent to one-hot and will not beat it. Only worth keeping if it lets you DROP the one-hot block for fewer dof. Enumerated because it is the canonical encoding method and the harness rejects it cheaply.

**Leakage:** Real risk if done naively: client label rates vary, so category means computed with same-client rows leak client-level label prevalence into that client's own features. The GroupKFold inner OOF above removes it. Never target-encode client_id (direct label leak, useless for unseen clients).

**Accept:** KEEP only if mean grouped-CV AUC delta >= +0.002 over the one-hot baseline, else REVERT.

### categorical_cross_features (priority: medium)

**How:** String-concat pairs then one-hot: content_type + '_' + main_intent, content_type + '_' + freshness_tier, main_intent + '_' + competition_level (fillna('missing') first). OneHotEncoder(handle_unknown='ignore', min_frequency=30) to prune rare crosses. ~20-35 extra columns.

**Why:** LR cannot express 'blog posts with transactional intent decline differently' without explicit crosses. Low cardinality keeps it tractable; moderate chance of a small gain.

**Leakage:** None; crosses of allowed static categoricals.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### groupwise_centering_cat_by_num (priority: medium)

**How:** Custom transformer: fit computes per-category means of selected numerics on the train fold (content_type mean of log_imp_prev, log_word_count; main_intent mean of prev_ctr); transform appends x - mean(x|category), unseen category -> global mean. 4-6 columns. (The per-client variant of this idea lives in client_aggregate_features.)

**Why:** Encodes categorical-by-numeric interaction as one column each instead of a full cross; 'this page has low CTR for its intent type' is more transferable than raw CTR. Modest expectations.

**Leakage:** None; means are label-free and fit on train folds only.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### kmeans_distance_features (priority: medium)

**How:** Pipeline branch: StandardScaler -> KMeans(n_clusters=8, n_init=10, random_state=0) fit on the numeric block of the train fold; append kmeans.transform(X) (8 distance columns) and optionally one-hot of predicted cluster. KMeans threads via OpenMP, unaffected by the n_jobs=1 joblib constraint. Fits in seconds at 22k x ~15.

**Why:** Gives LR access to 'page archetype' structure (e.g. old thin high-competition pages) it cannot form linearly. Hit rate on tabular is mediocre but nonzero; cheap enough to try.

**Leakage:** None when the KMeans is fit inside the CV pipeline; label-free.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT.

### nystroem_rbf_features (priority: medium)

**How:** sklearn.kernel_approximation.Nystroem(kernel='rbf', gamma=None, n_components=100, random_state=0) on the scaled numeric block, appended to the original linear features via FeatureUnion/ColumnTransformer; grid gamma in {0.05, 0.1, 0.5} if the first shot moves AUC. Alternative arm: RBFSampler(gamma=0.1, n_components=100, random_state=0), cheaper but cruder.

**Why:** Turns LR into an approximate kernel machine, capturing smooth multivariate nonlinearity that splines (univariate) miss. On weak-signal data the extra 100 dims often just add variance under grouped CV; genuine coin-flip, worth one cheap toggle.

**Leakage:** None; landmarks/components fit on train folds inside the pipeline.

**Accept:** Higher bar: KEEP only if mean grouped-CV AUC delta >= +0.003, else REVERT.

### pca_component_append (priority: low)

**How:** sklearn.decomposition.PCA(n_components=5, random_state=0) on the scaled numerics, components appended; TruncatedSVD(n_components=5) analogously on the one-hot block.

**Why:** Honest note: appending linear projections of features already in the design matrix cannot change LR's hypothesis space (span is unchanged) — expect exactly zero delta with LR. Only useful as INPUT to KMeans/Nystroem or a tree model (other dimensions). Enumerated for completeness; expect the harness to reject.

**Leakage:** None; fit on train folds inside the pipeline.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT (predicted outcome: revert).

### winsorize_clip_p99 (priority: low)

**How:** Custom transformer (or FunctionTransformer with train-fit quantiles): clip each raw numeric at its train-fold 1st/99th percentile before log1p/scaling.

**Why:** Redundant if log1p_all_skewed is kept (log already tames tails); can rescue the baseline if that method is rejected. Near-zero expected delta otherwise.

**Leakage:** None; clip thresholds fit on train folds.

**Accept:** KEEP if mean grouped-CV AUC delta >= +0.002, else REVERT; test after the log-transform decision.

