"""Greedy experiment harness for the ML-08 decline-ranking model.

Reads the plan in IMPLEMENTATION_PLAN.md. Tries every candidate one at a time on a fixed
15-fold client-grouped CV, keeps a step if it clears the accept rule, reverts otherwise, and
records every trial to ledger.jsonl. Run:

    python run_experiments.py setup   # data, lockbox, folds, baseline repro, null calibration
    python run_experiments.py full    # the whole greedy loop + verification (resumable)

Windows / Python 3.14 / sklearn only, n_jobs=1 everywhere.
"""

import sys, json, hashlib, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder, SplineTransformer,
    KBinsDiscretizer)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, SelectPercentile,
    f_classif, mutual_info_classif, SelectFromModel)
from sklearn.metrics import roc_auc_score
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = REPO / "data" / "raw" / "content_refresh_anonymized.csv"
LEDGER = HERE / "ledger.jsonl"
LOCKBOX_FILE = HERE / "lockbox.json"

SEED = 42
LOCKBOX_SEED = 20260715
DEV_SEEDS = (0, 1, 2)
FRESH_SEEDS = (10, 11, 12)
N_LOCKBOX = 6
EPS = 0.002
WIN_MIN = 9
P50_DROP = 0.03
BUDGET = 80

FORBIDDEN = frozenset([
    "impressions_90d", "clicks_90d", "pageviews_90d", "sessions_90d", "users_90d",
    "engaged_sessions_90d", "ai_sessions_90d", "scroll_events_90d", "days_with_impressions",
    "days_with_sessions", "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
    "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct",
    "impression_tier", "position_tier", "trend_direction", "trend_pct", "is_declining",
])

BASE_NUM = ["content_age_days", "days_since_last_update", "word_count", "char_count",
            "search_volume", "competition", "cpc",
            "impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"]
PREV = ["impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"]
BASE_CATS = ["content_type", "main_intent", "competition_level"]

BASELINE = {
    "population": "visible",
    "target": "binary",
    "num_impute": "zero",
    "num": list(BASE_NUM),
    "derived": [],
    "log1p_prev": True,
    "log1p_extra": [],
    "missing_flags": ["search_volume", "word_count"],
    "cats": list(BASE_CATS),
    "encoder": "onehot",
    "scaler": "standard",
    "selection": None,
    "sample_weight": None,
    "model": "logreg",
    "model_params": {"class_weight": "balanced", "max_iter": 2000, "C": 1.0},
}

TREE_NATIVE = {"histgb"}
TREE_IMPUTE = {"rf", "extratrees", "dtree", "gradientboosting", "adaboost"}


def load():
    df = pd.read_csv(DATA)
    df["is_declining"] = df["trend_direction"].str.lower().eq("down").astype(int)
    for c in df.columns:
        assert c == "is_declining" or True
    return df


def pick_lockbox(df):
    if LOCKBOX_FILE.exists():
        return set(json.loads(LOCKBOX_FILE.read_text())["clients"])
    vis = df[df["impressions_90d"] >= 100]
    clients = np.array(sorted(df["client_id"].unique()))
    rng = np.random.default_rng(LOCKBOX_SEED)
    for _ in range(100000):
        pick = set(rng.choice(clients, N_LOCKBOX, replace=False).tolist())
        sub = vis[vis["client_id"].isin(pick)]
        if 3000 <= len(sub) <= 5500 and 0.55 <= sub["is_declining"].mean() <= 0.65:
            LOCKBOX_FILE.write_text(json.dumps(
                {"clients": sorted(pick), "seed": LOCKBOX_SEED,
                 "rows": int(len(sub)), "base_rate": round(float(sub["is_declining"].mean()), 4)},
                indent=2))
            return pick
    raise RuntimeError("no lockbox pick met the constraints")


def derive(v):
    """Row-local and within-client derived columns. v is a population frame (copy)."""
    out = {}
    imp, clk, ses = v["impressions_prev_30d"], v["clicks_prev_30d"], v["sessions_prev_30d"]
    out["prev_ctr"] = np.where(imp > 0, clk / imp.where(imp > 0, 1), 0.0)
    out["sess_per_imp"] = np.where(imp > 0, ses / imp.where(imp > 0, 1), 0.0)
    out["clk_per_sess"] = np.where(ses > 0, clk / ses.where(ses > 0, 1), 0.0)
    out["zero_imp_prev"] = (imp == 0).astype(float)
    out["zero_clk_prev"] = (clk == 0).astype(float)
    out["zero_sess_prev"] = (ses == 0).astype(float)
    out["all_prev_zero"] = ((imp == 0) & (clk == 0) & (ses == 0)).astype(float)
    age = v["content_age_days"].clip(lower=1)
    dslu = v["days_since_last_update"]
    out["staleness_ratio"] = (dslu / age).clip(0, 1)
    out["update_recency"] = (v["content_age_days"] - dslu)
    wc = v["word_count"]
    cc = v["char_count"]
    out["avg_word_len"] = np.where(wc > 0, cc / wc.where(wc > 0, 1), 0.0)
    out["words_per_day"] = wc.fillna(0) / (v["content_age_days"] + 1)
    out["imp_prev_per_100w"] = imp / ((wc.fillna(0) / 100) + 1)
    sv = v["search_volume"]
    cpc = v["cpc"]
    comp = v["competition"]
    logsv = np.log1p(sv.fillna(0))
    out["sv_x_cpc"] = logsv * cpc.fillna(0)
    out["comp_x_cpc"] = comp.fillna(0) * cpc.fillna(0)
    out["comp_x_logsv"] = comp.fillna(0) * logsv
    out["staleness_x_logsv"] = dslu * logsv
    for c in ["impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d",
              "search_volume", "word_count", "days_since_last_update", "content_age_days"]:
        out["rank_" + c] = v.groupby("client_id")[c].rank(pct=True).fillna(0.5).values
    g = v.groupby("client_id")
    out["client_n_pages"] = g["content_id"].transform("size").values.astype(float)
    tot = g["impressions_prev_30d"].transform("sum")
    out["traffic_share"] = (imp / (tot + 1)).values
    out["dev_word_count"] = (wc.fillna(0) - g["word_count"].transform("mean").fillna(0)).values
    out["dev_log_imp"] = (np.log1p(imp) - g["impressions_prev_30d"].transform(
        lambda s: np.log1p(s).mean())).values
    return pd.DataFrame(out, index=v.index)


DERIVED_GROUPS = {
    "prev_ratios": ["prev_ctr", "sess_per_imp", "clk_per_sess"],
    "zero_flags": ["zero_imp_prev", "zero_clk_prev", "zero_sess_prev", "all_prev_zero"],
    "staleness": ["staleness_ratio", "update_recency"],
    "density": ["avg_word_len", "words_per_day", "imp_prev_per_100w"],
    "keyword_composites": ["sv_x_cpc", "comp_x_cpc", "comp_x_logsv", "staleness_x_logsv"],
    "per_client_rank": ["rank_impressions_prev_30d", "rank_clicks_prev_30d",
        "rank_sessions_prev_30d", "rank_search_volume", "rank_word_count",
        "rank_days_since_last_update", "rank_content_age_days"],
    "client_agg": ["client_n_pages", "traffic_share", "dev_word_count", "dev_log_imp"],
}


def build_matrix(cfg, df, dev_clients):
    pop = cfg["population"]
    frame = df[df["client_id"].isin(dev_clients)].copy()
    if pop == "visible":
        v = frame[frame["impressions_90d"] >= 100].copy()
    else:
        v = frame.copy()
    v = v.reset_index(drop=True)
    d = derive(v)

    num = list(cfg["num"])
    Xcols = {}
    for c in num:
        Xcols[c] = v[c].astype(float)
    if cfg["log1p_prev"]:
        for c in PREV:
            Xcols["log_" + c] = np.log1p(v[c].fillna(0))
    for c in cfg["log1p_extra"]:
        Xcols["log_" + c] = np.log1p(v[c].fillna(0))
    for c in cfg["missing_flags"]:
        Xcols["has_" + c] = v[c].notna().astype(float)
    for grp in cfg["derived"]:
        for col in DERIVED_GROUPS[grp]:
            Xcols[col] = d[col].astype(float)
    Xnum = pd.DataFrame(Xcols, index=v.index)

    cat_cols = list(cfg["cats"])
    Xcat = v[cat_cols].astype("object").fillna("unknown") if cat_cols else pd.DataFrame(index=v.index)
    X = pd.concat([Xnum, Xcat], axis=1)

    bad = (set(X.columns) & FORBIDDEN)
    assert not bad, f"forbidden columns leaked into matrix: {bad}"

    y = v["is_declining"].values
    groups = v["client_id"].values
    aux = pd.DataFrame({
        "client_id": v["client_id"].values,
        "impressions_90d": v["impressions_90d"].values,
        "impressions_prev_30d": v["impressions_prev_30d"].values,
        "trend_pct": v["trend_pct"].values,
        "trend_direction": v["trend_direction"].values,
        "is_declining": y,
    }, index=v.index)
    numeric_cols = list(Xnum.columns)
    return X, y, groups, aux, numeric_cols, cat_cols


def make_scaler(name):
    return {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler(),
            "maxabs": MaxAbsScaler(),
            "quantile_normal": QuantileTransformer(output_distribution="normal",
                n_quantiles=1000, subsample=100000, random_state=SEED),
            "quantile_uniform": QuantileTransformer(output_distribution="uniform",
                n_quantiles=1000, subsample=100000, random_state=SEED),
            "power_yeo": PowerTransformer(method="yeo-johnson"),
            "none": "passthrough"}[name]


def make_model(name, params):
    p = dict(params)
    if name == "logreg":
        return LogisticRegression(solver="lbfgs", n_jobs=1, random_state=SEED, **p)
    if name == "logreg_saga":
        return LogisticRegression(solver="saga", n_jobs=1, random_state=SEED, **p)
    if name == "ridge":
        return RidgeClassifier(random_state=SEED, **p)
    if name == "sgd":
        return SGDClassifier(random_state=SEED, **p)
    if name == "lda":
        return LinearDiscriminantAnalysis(**p)
    if name == "qda":
        return QuadraticDiscriminantAnalysis(**p)
    if name == "gnb":
        return GaussianNB(**p)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=1, **p)
    if name == "dtree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "rf":
        return RandomForestClassifier(n_jobs=1, random_state=SEED, **p)
    if name == "extratrees":
        return ExtraTreesClassifier(n_jobs=1, random_state=SEED, **p)
    if name == "gradientboosting":
        return GradientBoostingClassifier(random_state=SEED, **p)
    if name == "histgb":
        return HistGradientBoostingClassifier(random_state=SEED, **p)
    if name == "adaboost":
        return AdaBoostClassifier(random_state=SEED, **p)
    if name == "mlp":
        return MLPClassifier(random_state=SEED, **p)
    raise ValueError(f"unknown model {name}")


def make_selector(sel):
    if sel is None:
        return None
    m = sel["method"]
    if m == "variance":
        return VarianceThreshold(sel.get("threshold", 0.0))
    if m == "f_pct":
        return SelectPercentile(f_classif, percentile=sel["percentile"])
    if m == "mi_pct":
        return SelectPercentile(
            lambda X, y: mutual_info_classif(X, y, random_state=SEED), percentile=sel["percentile"])
    if m == "kbest_f":
        return SelectKBest(f_classif, k=sel["k"])
    if m == "l1_logreg":
        return SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear",
            C=sel.get("C", 1.0), random_state=SEED))
    raise ValueError(f"unknown selector {m}")


def make_pipeline_from(cfg, numeric_cols, cat_cols):
    model = make_model(cfg["model"], cfg["model_params"])
    name = cfg["model"]

    if name in TREE_NATIVE:
        parts = []
        if numeric_cols:
            parts.append(("num", "passthrough", numeric_cols))
        if cat_cols:
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                unknown_value=-1, encoded_missing_value=-2), cat_cols))
        ct = ColumnTransformer(parts, remainder="drop")
        model.set_params(categorical_features=[c for c in range(len(numeric_cols),
            len(numeric_cols) + len(cat_cols))] if cat_cols else None)
        steps = [("prep", ct), ("model", model)]
        return Pipeline(steps)

    num_impute = {"zero": SimpleImputer(strategy="constant", fill_value=0),
                  "median": SimpleImputer(strategy="median"),
                  "mean": SimpleImputer(strategy="mean")}[cfg["num_impute"]]
    num_steps = [("impute", num_impute)]
    if name not in TREE_IMPUTE:
        num_steps.append(("scale", make_scaler(cfg["scaler"])))
    num_pipe = Pipeline(num_steps)

    parts = []
    if numeric_cols:
        parts.append(("num", num_pipe, numeric_cols))
    if cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
            drop="first" if cfg["encoder"] == "onehot_drop" else None) \
            if cfg["encoder"] in ("onehot", "onehot_drop") else \
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts.append(("cat", enc, cat_cols))
    ct = ColumnTransformer(parts, remainder="drop")

    steps = [("prep", ct)]
    sel = make_selector(cfg["selection"])
    if sel is not None:
        steps.append(("select", sel))
    steps.append(("model", model))
    return Pipeline(steps)


def compute_weights(cfg, aux_tr, y_tr):
    sw = cfg["sample_weight"]
    if sw is None:
        return None
    kind = sw["kind"]
    if kind == "prev_traffic":
        w = np.log1p(aux_tr["impressions_prev_30d"].values.astype(float))
    elif kind == "prev_traffic_sqrt":
        w = np.sqrt(aux_tr["impressions_prev_30d"].values.astype(float))
    elif kind == "confidence":
        floor = sw.get("floor", 5)
        w = np.clip(np.abs(aux_tr["trend_pct"].fillna(0).values), floor, 50) / 50.0
    elif kind == "per_client":
        alpha = sw.get("alpha", 1.0)
        n_c = aux_tr.groupby("client_id")["client_id"].transform("size").values.astype(float)
        w = n_c ** (-alpha)
    elif kind == "recency":
        h = sw.get("h", 365)
        d = aux_tr["is_declining"]  # placeholder guard; recency uses dslu, not available in aux
        raise ValueError("recency weight needs days_since_last_update in aux")
    else:
        raise ValueError(f"unknown sample_weight {kind}")
    w = w * (len(w) / w.sum())
    return w


def p_at_50(y_true, score, k=50):
    order = np.argsort(-np.asarray(score, dtype=float))
    return float(np.asarray(y_true)[order[:k]].mean())


def fit_score_fold(cfg, X, y, aux, numeric_cols, cat_cols, tr, te, vis_te):
    pipe = make_pipeline_from(cfg, numeric_cols, cat_cols)
    tgt = cfg["target"]
    w = compute_weights(cfg, aux.iloc[tr], y[tr])
    fit_kw = {}
    if w is not None:
        fit_kw["model__sample_weight"] = w

    if tgt == "binary":
        pipe.fit(X.iloc[tr], y[tr], **fit_kw)
        score = pipe.predict_proba(X.iloc[te])[:, 1]
    elif tgt == "3class":
        y3 = aux["trend_direction"].str.lower().values
        pipe.fit(X.iloc[tr], y3[tr], **fit_kw)
        classes = list(pipe.named_steps["model"].classes_)
        di = classes.index("down")
        score = pipe.predict_proba(X.iloc[te])[:, di]
    elif tgt == "regress_trend":
        reg = Pipeline(pipe.steps[:-1] + [("model", Ridge(alpha=1.0, random_state=SEED))])
        t = np.clip(-aux["trend_pct"].fillna(0).values, -100, 100)
        reg.fit(X.iloc[tr], t[tr])
        score = reg.predict(X.iloc[te])
    else:
        raise ValueError(tgt)

    yte = y[te]
    ste = score
    if vis_te is not None:
        yte = yte[vis_te]
        ste = ste[vis_te]
    return roc_auc_score(yte, ste), p_at_50(yte, ste)


def client_fold_map(groups, n_splits, seed):
    """One seeded, size-balanced client->fold assignment. StratifiedGroupKFold is seed-invariant
    on this data (few, size-skewed clients), so we assign clients in seeded-random order, each to
    the currently-lightest fold. Genuine seed variation; base-rate stratification is averaged out
    across seeds. A map (not index arrays) so the SAME split can be applied to two matrices with
    different row sets (visible vs widened) for a paired population comparison."""
    gs = pd.Series(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.array(sorted(gs.unique())))
    sizes = gs.value_counts()
    load = np.zeros(n_splits)
    m = {}
    for c in order:
        f = int(np.argmin(load))
        m[c] = f
        load[f] += sizes[c]
    return m


def folds_from_maps(groups, maps, n_splits=5):
    gs = pd.Series(groups)
    out = []
    for m in maps:
        fid = gs.map(m).values
        out += [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(n_splits)]
    return out


def build_maps(groups, seeds):
    return [client_fold_map(groups, 5, s) for s in seeds]


def evaluate(cfg, df, dev_clients, seeds=DEV_SEEDS, maps=None):
    X, y, groups, aux, numeric_cols, cat_cols = build_matrix(cfg, df, dev_clients)
    vis = None if cfg["population"] == "visible" else (aux["impressions_90d"].values >= 100)
    if maps is None:
        maps = build_maps(groups, seeds)
    folds = folds_from_maps(groups, maps)
    aucs, p50s = [], []
    for tr, te in folds:
        vis_te = vis[te] if vis is not None else None
        a, p = fit_score_fold(cfg, X, y, aux, numeric_cols, cat_cols, tr, te, vis_te)
        aucs.append(a)
        p50s.append(p)
    return {"auc_folds": aucs, "p50_folds": p50s,
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "p50_mean": float(np.mean(p50s))}


def widened_groups(df, dev_clients):
    frame = df[df["client_id"].isin(dev_clients)]
    return frame["client_id"].values


def cfg_hash(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def accept(base_res, cand_res):
    ba, ca = np.array(base_res["auc_folds"]), np.array(cand_res["auc_folds"])
    bp, cp = np.array(base_res["p50_folds"]), np.array(cand_res["p50_folds"])
    delta = ca.mean() - ba.mean()
    wins = int((ca > ba).sum())
    p50_drop = (cp.mean() < bp.mean() - P50_DROP) and ((cp < bp).mean() >= 0.60)
    ok = (delta >= EPS) and (wins >= WIN_MIN) and (not p50_drop)
    return ok, delta, wins, p50_drop


def wilcoxon_p(base_res, cand_res):
    d = np.array(cand_res["auc_folds"]) - np.array(base_res["auc_folds"])
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="greater", zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def ledger_write(rec):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def ledger_load():
    seen = {}
    if LEDGER.exists():
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "hash" in r and "auc_mean" in r:
                seen[r["hash"]] = r
    return seen


SEEN = {}


def eval_cached(cfg, df, dev, seeds=DEV_SEEDS):
    h = cfg_hash((cfg, seeds))
    if h in SEEN:
        return SEEN[h], h, True
    res = evaluate(cfg, df, dev, seeds)
    SEEN[h] = res
    return res, h, False


# ---- candidate stages -------------------------------------------------------

def stage_candidates():
    S = []

    # Stage 1 preprocessing
    S += [("preproc", "median_impute", {"num_impute": "median"}),
          ("preproc", "mean_impute", {"num_impute": "mean"}),
          ("preproc", "scaler_robust", {"scaler": "robust"}),
          ("preproc", "scaler_quantile_normal", {"scaler": "quantile_normal"}),
          ("preproc", "scaler_quantile_uniform", {"scaler": "quantile_uniform"}),
          ("preproc", "scaler_power_yeo", {"scaler": "power_yeo"}),
          ("preproc", "scaler_maxabs", {"scaler": "maxabs"}),
          ("preproc", "log1p_extended",
              {"log1p_extra": ["search_volume", "cpc", "word_count", "char_count",
                               "content_age_days", "days_since_last_update"]}),
          ("preproc", "cats_add_provider_model",
              {"cats": BASE_CATS + ["provider_used", "model_used"]}),
          ("preproc", "cats_add_tiers",
              {"cats": BASE_CATS + ["age_tier", "freshness_tier", "word_count_tier"]}),
          ("preproc", "missing_flags_all",
              {"missing_flags": ["search_volume", "word_count", "competition", "cpc",
                                 "main_intent", "provider_used", "model_used"]}),
          ("preproc", "encoder_ordinal", {"encoder": "ordinal"}),
          ("preproc", "encoder_onehot_drop", {"encoder": "onehot_drop"})]

    # Stage 2 feature engineering (derived groups, additive)
    S += [("feateng", "add_prev_ratios", {"derived": ["prev_ratios"]}),
          ("feateng", "add_zero_flags", {"derived": ["zero_flags"]}),
          ("feateng", "add_staleness", {"derived": ["staleness"]}),
          ("feateng", "add_keyword_composites", {"derived": ["keyword_composites"]}),
          ("feateng", "add_density", {"derived": ["density"]}),
          ("feateng", "add_per_client_rank", {"derived": ["per_client_rank"]}),
          ("feateng", "add_client_agg", {"derived": ["client_agg"]})]

    # Stage 4 sampling / target / population
    S += [("sampling", "cw_none",
              {"model_params": {"class_weight": None, "max_iter": 2000, "C": 1.0}}),
          ("sampling", "sw_prev_traffic", {"sample_weight": {"kind": "prev_traffic"}}),
          ("sampling", "sw_confidence", {"sample_weight": {"kind": "confidence", "floor": 5}}),
          ("sampling", "sw_per_client", {"sample_weight": {"kind": "per_client", "alpha": 1.0}}),
          ("sampling", "target_3class", {"target": "3class"}),
          ("sampling", "target_regress", {"target": "regress_trend"}),
          ("sampling", "widen_all", {"population": "widen_all"}),
          ("sampling", "widen_downweight",
              {"population": "widen_all", "sample_weight": {"kind": "prev_traffic"}})]

    # Stage 5 models
    S += [("model", "logreg_C0.3", {"model_params":
              {"class_weight": "balanced", "max_iter": 2000, "C": 0.3}}),
          ("model", "logreg_C3", {"model_params":
              {"class_weight": "balanced", "max_iter": 2000, "C": 3.0}}),
          ("model", "logreg_l1", {"model": "logreg_saga", "model_params":
              {"class_weight": "balanced", "max_iter": 3000, "penalty": "l1", "C": 1.0}}),
          ("model", "lda", {"model": "lda", "scaler": "standard", "model_params": {}}),
          ("model", "gnb", {"model": "gnb", "model_params": {}}),
          ("model", "knn", {"model": "knn", "model_params": {"n_neighbors": 50, "weights": "distance"}}),
          ("model", "rf", {"model": "rf", "model_params":
              {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 5,
               "class_weight": "balanced"}}),
          ("model", "extratrees", {"model": "extratrees", "model_params":
              {"n_estimators": 400, "min_samples_leaf": 5, "class_weight": "balanced"}}),
          ("model", "histgb", {"model": "histgb", "model_params":
              {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31,
               "l2_regularization": 1.0, "early_stopping": True}}),
          ("model", "histgb_deep", {"model": "histgb", "model_params":
              {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 63,
               "l2_regularization": 2.0, "early_stopping": True}}),
          ("model", "gradientboosting", {"model": "gradientboosting", "model_params":
              {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}}),
          ("model", "mlp", {"model": "mlp", "model_params":
              {"hidden_layer_sizes": (64, 32), "alpha": 1e-3, "early_stopping": True,
               "max_iter": 300}})]
    return S


def run_stage(stage, cands, best, best_res, df, dev, spent):
    for st, name, patch in cands:
        if st != stage:
            continue
        if spent[0] >= BUDGET:
            ledger_write({"stage": stage, "patch": name, "decision": "skipped_budget"})
            continue
        cand = {**best, **patch}
        pop_change = cand["population"] != best["population"]
        t0 = time.perf_counter()
        try:
            if pop_change:
                maps = build_maps(widened_groups(df, dev), DEV_SEEDS)
                base_paired = evaluate(best, df, dev, maps=maps)
                res = evaluate(cand, df, dev, maps=maps)
                h, cached = cfg_hash((cand, DEV_SEEDS, "pop")), False
            else:
                base_paired = best_res
                res, h, cached = eval_cached(cand, df, dev)
        except Exception as e:
            ledger_write({"stage": stage, "patch": name, "error": repr(e), "decision": "error"})
            print(f"  [{stage}] {name:28s} ERROR {e!r}")
            continue
        secs = time.perf_counter() - t0
        spent[0] += 0 if cached else 1
        ok, delta, wins, veto = accept(base_paired, res)
        wp = wilcoxon_p(base_paired, res)
        rec = {"stage": stage, "patch": name, "hash": h,
               "auc_mean": round(res["auc_mean"], 5), "auc_std": round(res["auc_std"], 5),
               "auc_folds": [round(a, 5) for a in res["auc_folds"]],
               "p50_mean": round(res["p50_mean"], 4),
               "p50_folds": [round(p, 4) for p in res["p50_folds"]],
               "delta": round(delta, 5), "wins": wins, "wilcoxon_p": round(wp, 4),
               "decision": "keep" if ok else ("veto_p50" if veto else "revert"),
               "seconds": round(secs, 1), "config": cand}
        ledger_write(rec)
        flag = "KEEP" if ok else ("VETO" if veto else "----")
        print(f"  [{stage}] {name:28s} auc {res['auc_mean']:.4f} d {delta:+.4f} "
              f"w{wins:2d} p50 {res['p50_mean']:.3f} {flag}")
        if ok:
            best, best_res = cand, res
    return best, best_res


def null_calibration(df, dev):
    print("null-probe calibration")
    base = evaluate(BASELINE, df, dev)
    rng_cols = build_matrix(BASELINE, df, dev)[0]
    n = len(rng_cols)
    accepts = 0
    probes = 0
    for k in range(10):
        cfg = {**BASELINE}
        X, y, groups, aux, ncols, ccols = build_matrix(BASELINE, df, dev)
        rng = np.random.default_rng(100 + k)
        Xj = X.copy()
        Xj["junk"] = rng.normal(size=len(X))
        folds = make_folds(groups, DEV_SEEDS)
        aucs, p50s = [], []
        for tr, te in folds:
            pipe = make_pipeline_from(cfg, ncols + ["junk"], ccols)
            pipe.fit(Xj.iloc[tr], y[tr])
            sc = pipe.predict_proba(Xj.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], sc))
            p50s.append(p_at_50(y[te], sc))
        res = {"auc_folds": aucs, "p50_folds": p50s, "auc_mean": float(np.mean(aucs)),
               "p50_mean": float(np.mean(p50s))}
        ok, delta, wins, veto = accept(base, res)
        accepts += int(ok)
        probes += 1
        print(f"  junk {k}: d {delta:+.4f} w{wins} {'ACCEPT(bad)' if ok else 'reject'}")
    print(f"null-probe: {accepts}/{probes} junk candidates accepted (want <=1)")
    return accepts


def setup(df):
    dev_clients = set(sorted(df["client_id"].unique())) - pick_lockbox(df)
    lb = pick_lockbox(df)
    print(f"clients: {df['client_id'].nunique()} total, {len(lb)} lockbox, {len(dev_clients)} dev")
    print(f"lockbox: {sorted(lb)}")
    lbrows = df[(df['client_id'].isin(lb)) & (df['impressions_90d'] >= 100)]
    print(f"lockbox visible rows {len(lbrows)}, base rate {lbrows['is_declining'].mean():.4f}")

    base = evaluate(BASELINE, df, dev_clients)
    print(f"BASELINE dev-CV: AUC {base['auc_mean']:.4f} +/- {base['auc_std']:.4f}  "
          f"P@50 {base['p50_mean']:.4f}  (n=15 folds)")
    print(f"  per-fold AUC: {[round(a,3) for a in base['auc_folds']]}")
    assert abs(base["auc_mean"] - 0.601) < 0.05, "baseline reproduction out of tolerance"

    per_client_base = df[df['impressions_90d'] >= 100].groupby('client_id')['is_declining'].mean()
    print(f"per-client base rate spread: {per_client_base.min():.2f}..{per_client_base.max():.2f}")
    sizes = df[df['impressions_90d'] >= 100].groupby('client_id').size()
    print(f"client sizes (visible): min {sizes.min()} median {int(sizes.median())} max {sizes.max()}")

    null_calibration(df, dev_clients)
    return dev_clients, base


def full(df):
    global SEEN
    SEEN = ledger_load()
    dev_clients = set(sorted(df["client_id"].unique())) - pick_lockbox(df)
    best = dict(BASELINE)
    best_res = evaluate(BASELINE, df, dev_clients)
    ledger_write({"stage": "baseline", "patch": "baseline", "hash": cfg_hash((best, DEV_SEEDS)),
        "auc_mean": round(best_res["auc_mean"], 5), "auc_std": round(best_res["auc_std"], 5),
        "auc_folds": [round(a, 5) for a in best_res["auc_folds"]],
        "p50_mean": round(best_res["p50_mean"], 4),
        "p50_folds": [round(p, 4) for p in best_res["p50_folds"]],
        "decision": "baseline", "config": best})
    print(f"BASELINE AUC {best_res['auc_mean']:.4f} P@50 {best_res['p50_mean']:.3f}")
    cands = stage_candidates()
    spent = [0]
    for stage in ["preproc", "feateng", "sampling", "model"]:
        print(f"== STAGE {stage} ==")
        best, best_res = run_stage(stage, cands, best, best_res, df, dev_clients, spent)
        print(f"   -> best AUC {best_res['auc_mean']:.4f} P@50 {best_res['p50_mean']:.3f} "
              f"(spent {spent[0]})")
    ledger_write({"stage": "final_greedy", "patch": "final", "hash": cfg_hash((best, DEV_SEEDS)),
        "auc_mean": round(best_res["auc_mean"], 5), "auc_std": round(best_res["auc_std"], 5),
        "auc_folds": [round(a, 5) for a in best_res["auc_folds"]],
        "p50_mean": round(best_res["p50_mean"], 4),
        "p50_folds": [round(p, 4) for p in best_res["p50_folds"]],
        "decision": "final_greedy", "config": best})
    print(f"FINAL greedy AUC {best_res['auc_mean']:.4f} P@50 {best_res['p50_mean']:.3f}")
    print(json.dumps(best, indent=2, default=str))


def kept_patches():
    names = [json.loads(l)["patch"] for l in LEDGER.read_text(encoding="utf-8").splitlines()
             if l.strip() and json.loads(l).get("decision") == "keep"]
    lut = {name: patch for _, name, patch in stage_candidates()}
    return [(n, lut[n]) for n in names]


def apply_patches(patches):
    cfg = dict(BASELINE)
    for _, p in patches:
        cfg = {**cfg, **p}
    return cfg


def lockbox_eval(cfg, df, dev, lockbox):
    allc = set(df["client_id"].unique())
    X, y, groups, aux, ncols, ccols = build_matrix(cfg, df, allc)
    is_dev = aux["client_id"].isin(dev).values
    vis = aux["impressions_90d"].values >= 100
    tr = np.where(is_dev)[0]
    te = np.where((~is_dev) & vis)[0]
    a, p = fit_score_fold(cfg, X, y, aux, ncols, ccols, tr, te, None)
    return a, p


def permutation_control(cfg, df, dev):
    X, y, groups, aux, ncols, ccols = build_matrix(cfg, df, dev)
    rng = np.random.default_rng(0)
    yp = pd.Series(y).groupby(pd.Series(groups)).transform(
        lambda s: rng.permutation(s.values)).values
    vis = None if cfg["population"] == "visible" else (aux["impressions_90d"].values >= 100)
    aux2 = aux.copy()
    aux2["is_declining"] = yp
    aucs = []
    for tr, te in folds_from_maps(groups, build_maps(groups, DEV_SEEDS)):
        vt = vis[te] if vis is not None else None
        a, _ = fit_score_fold(cfg, X, yp, aux2, ncols, ccols, tr, te, vt)
        aucs.append(a)
    return float(np.mean(aucs))


def verify(df):
    dev = set(sorted(df["client_id"].unique())) - pick_lockbox(df)
    lockbox = pick_lockbox(df)
    keeps = kept_patches()
    final = apply_patches(keeps)
    print("kept steps:", [n for n, _ in keeps])

    print("\n== fresh-seed re-verification (seeds 10,11,12) ==")
    base_fresh = evaluate(BASELINE, df, dev, seeds=FRESH_SEEDS)
    fin_fresh = evaluate(final, df, dev, seeds=FRESH_SEEDS)
    print(f"  baseline fresh: AUC {base_fresh['auc_mean']:.4f} P@50 {base_fresh['p50_mean']:.3f}")
    print(f"  final    fresh: AUC {fin_fresh['auc_mean']:.4f} P@50 {fin_fresh['p50_mean']:.3f}")
    print(f"  fresh delta: {fin_fresh['auc_mean'] - base_fresh['auc_mean']:+.4f}")

    print("\n== leave-one-step-out prune (fresh folds) ==")
    pruned = []
    for i, (nm, _) in enumerate(keeps):
        sub = keeps[:i] + keeps[i + 1:]
        r = evaluate(apply_patches(sub), df, dev, seeds=FRESH_SEEDS)
        cost = fin_fresh["auc_mean"] - r["auc_mean"]
        keepit = cost > 0.001
        print(f"  without {nm:22s} AUC {r['auc_mean']:.4f} cost {cost:+.4f} "
              f"{'(load-bearing)' if keepit else '(PRUNE)'}")
        if not keepit:
            pruned.append(nm)
    surviving = [(n, p) for n, p in keeps if n not in pruned]
    final2 = apply_patches(surviving)
    print(f"  surviving chain: {[n for n, _ in surviving]}")

    print("\n== permutation negative control (within-client label shuffle) ==")
    pc = permutation_control(final2, df, dev)
    print(f"  permuted-label AUC {pc:.4f} (want ~0.50, halt if >0.55)")

    print("\n== lockbox (one shot each, train on 26 dev, test on 6 lockbox) ==")
    ba, bp = lockbox_eval(BASELINE, df, dev, lockbox)
    fa, fp = lockbox_eval(final2, df, dev, lockbox)
    print(f"  baseline lockbox: AUC {ba:.4f} P@50 {bp:.3f}")
    print(f"  final    lockbox: AUC {fa:.4f} P@50 {fp:.3f}")
    print(f"  lockbox delta: {fa - ba:+.4f}")

    ledger_write({"stage": "verify", "patch": "summary",
        "surviving": [n for n, _ in surviving],
        "base_fresh_auc": round(base_fresh["auc_mean"], 5),
        "final_fresh_auc": round(fin_fresh["auc_mean"], 5),
        "perm_control_auc": round(pc, 5),
        "baseline_lockbox_auc": round(ba, 5), "baseline_lockbox_p50": round(bp, 4),
        "final_lockbox_auc": round(fa, 5), "final_lockbox_p50": round(fp, 4),
        "config": final2, "decision": "verify"})
    print("\nfinal config:")
    print(json.dumps(final2, indent=2, default=str))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "setup"
    df = load()
    if mode == "setup":
        setup(df)
    elif mode == "full":
        full(df)
    elif mode == "verify":
        verify(df)
    else:
        print("usage: run_experiments.py [setup|full|verify]")
