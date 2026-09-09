"""
Spectrum quality scoring.

Two modes:

  rule    A transparent composite of the structural features. No training
          data needed, works on the first file you ever process, and is the
          right thing to ship first.

  model   A gradient-boosted classifier trained on your own data. Trained on
          labels that come from a PERMISSIVE search union, not a single
          closed search, to avoid teaching the model to reject exactly the
          spectra you want to rescue.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Features used by the trained model. Deliberately excludes raw intensity
# scale (tic, base_peak_intensity) so the model cannot just learn "bright".
MODEL_FEATURES = [
    "n_peaks_above_noise", "snr_proxy", "entropy", "norm_entropy",
    "top10_frac", "top20_frac", "log_dynamic_range",
    "frac_above_precursor", "n_residue_gaps", "gap_density",
    "longest_tag", "n_tags_ge2", "n_tags_ge3", "n_tags_ge4",
    "n_complementary", "complementary_tic_frac",
    "n_isotope_clusters", "isotope_tic_frac",
    "loss_h2o_frac", "loss_nh3_frac",
    "isolation_purity", "n_cofragmented", "charge",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rule_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite quality score in [0, 1]. Weights are hand-set from what is known
    to separate interpretable from uninterpretable spectra. They are a starting
    point, not a calibrated model - retrain once you have labels.
    """
    d = df

    # sequence-tag evidence: the strongest single structural signal
    tag = np.clip(d["longest_tag"].fillna(0) / 6.0, 0, 1)
    tag3 = np.clip(d["n_tags_ge3"].fillna(0) / 8.0, 0, 1)

    # complementarity: b/y pairs summing to the precursor
    comp = np.clip(d["n_complementary"].fillna(0) / 8.0, 0, 1)
    comp_int = np.clip(d["complementary_tic_frac"].fillna(0) / 0.3, 0, 1)

    # deisotoping success
    iso = np.clip(d["isotope_tic_frac"].fillna(0) / 0.5, 0, 1)

    # peak richness, saturating so very dense noise spectra do not win
    peaks = np.clip(d["n_peaks_above_noise"].fillna(0) / 40.0, 0, 1)

    # concentration of signal: pure noise is flat, so low normalised entropy
    # is good, but an extremely spiky spectrum (one peak) is also bad
    ne = d["norm_entropy"].fillna(1.0)
    conc = np.clip(1.0 - np.abs(ne - 0.75) / 0.4, 0, 1)

    # dynamic range above the noise floor
    dyn = np.clip(d["log_dynamic_range"].fillna(0) / 2.0, 0, 1)

    # precursor cleanliness; missing purity is treated as neutral
    purity = d["isolation_purity"].fillna(0.6)

    raw = (
        2.2 * tag
        + 1.2 * tag3
        + 1.4 * comp
        + 1.0 * comp_int
        + 1.2 * iso
        + 1.0 * peaks
        + 0.8 * conc
        + 0.6 * dyn
        + 0.8 * purity
    )
    # centre so that a typical mediocre spectrum lands near 0.5
    return pd.Series(_sigmoid((raw - 4.6) * 1.3), index=d.index).clip(0, 1)


def build_labels(df: pd.DataFrame,
                 denovo_score_col: str = "denovo_score",
                 denovo_threshold: float = 0.9) -> pd.Series:
    """
    Training labels.

      1 = interpretable: identified by ANY available evidence
      0 = uninterpretable: missed by everything
     -1 = held out: do not train on these

    The held-out class is the point. Spectra that only one method identifies,
    or that look structurally strong but are unassigned, must not be labelled
    negative or the model learns to discard the rescue pile.
    """
    label = pd.Series(0, index=df.index, dtype=int)

    assigned = df.get("assigned", pd.Series(False, index=df.index)).fillna(False)
    label[assigned.astype(bool)] = 1

    if denovo_score_col in df.columns:
        strong_denovo = df[denovo_score_col].fillna(-1) >= denovo_threshold
        label[strong_denovo & ~assigned.astype(bool)] = 1

    # hold out structurally strong but unassigned spectra
    strong = (df["longest_tag"].fillna(0) >= 3) & (df["n_complementary"].fillna(0) >= 2)
    label[(label == 0) & strong] = -1

    return label


def train_model(df: pd.DataFrame, out_path: str,
                group_col: str = "run_id", seed: int = 0):
    """
    Train a gradient-boosted classifier. Splits by run so the same peptide
    cannot appear on both sides of the split - a random split leaks badly and
    gives a fake AUC near 0.99.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise SystemExit(
            "scikit-learn is required to train. Run: pip install scikit-learn")

    # LightGBM if available, otherwise sklearn's histogram gradient booster.
    # Both handle NaN natively, which matters because half these features are
    # legitimately undefined on sparse spectra.
    try:
        import lightgbm as lgb
        make_model = lambda: lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1)
        backend = "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        make_model = lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
        backend = "sklearn_hist_gbdt"

    labels = build_labels(df)
    train = df[labels >= 0].copy()
    y = labels[labels >= 0]

    if y.nunique() < 2:
        raise SystemExit("Need both positive and negative examples to train.")

    feats = [c for c in MODEL_FEATURES if c in train.columns]
    X = train[feats].replace([np.inf, -np.inf], np.nan)

    runs = train[group_col].unique() if group_col in train else np.array(["all"])
    rng = np.random.default_rng(seed)
    runs = np.asarray(runs, dtype=object)
    rng.shuffle(runs)
    n_hold = max(1, len(runs) // 4)
    holdout = set(runs[:n_hold]) if len(runs) > 1 else set()

    is_val = train[group_col].isin(holdout) if holdout else pd.Series(
        rng.random(len(train)) < 0.25, index=train.index)

    model = make_model()
    model.fit(X[~is_val], y[~is_val])

    report = {"backend": backend, "features": feats,
              "n_train": int((~is_val).sum()), "n_val": int(is_val.sum()),
              "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
              "n_held_out": int((labels == -1).sum())}
    if is_val.sum() > 20 and y[is_val].nunique() == 2:
        pred = model.predict_proba(X[is_val])[:, 1]
        report["val_auc"] = float(roc_auc_score(y[is_val], pred))

    # single-feature baselines, so you can see whether the model earns its keep
    baselines = {}
    for c in ["longest_tag", "n_peaks_above_noise", "entropy", "n_complementary"]:
        if c in X.columns and y.nunique() == 2:
            v = X[c].fillna(X[c].median())
            try:
                baselines[c] = float(roc_auc_score(y, v))
            except ValueError:
                pass
    report["single_feature_auc"] = baselines
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        report["feature_importance"] = dict(
            sorted(zip(feats, [float(v) for v in importance]),
                   key=lambda kv: -kv[1]))
    elif is_val.sum() > 20 and y[is_val].nunique() == 2:
        # HistGradientBoosting has no native importances; permutation
        # importance on the held-out runs is the honest substitute.
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(
            model, X[is_val].fillna(X[is_val].median()), y[is_val],
            n_repeats=5, random_state=seed, scoring="roc_auc")
        report["feature_importance"] = dict(
            sorted(zip(feats, [float(v) for v in pi.importances_mean]),
                   key=lambda kv: -kv[1]))

    import pickle
    with open(out_path, "wb") as fh:
        pickle.dump({"model": model, "features": feats}, fh)
    with open(os.path.splitext(out_path)[0] + ".report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    if "val_auc" in report and baselines:
        best_single = max(baselines.values())
        if report["val_auc"] - best_single < 0.03:
            print("\n  [note] the model beats the best single feature by less "
                  "than 0.03 AUC. The ML layer is not earning its complexity; "
                  "consider staying with --scorer rule.")
    return model


def apply_model(df: pd.DataFrame, model_path: str) -> pd.Series:
    import pickle
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)
    X = df[bundle["features"]].replace([np.inf, -np.inf], np.nan)
    return pd.Series(bundle["model"].predict_proba(X)[:, 1], index=df.index)


def add_quality_score(df: pd.DataFrame, scorer: str = "rule",
                      model_path: str | None = None) -> pd.DataFrame:
    df = df.copy()
    if scorer == "model" and model_path and os.path.exists(model_path):
        df["qc_score"] = apply_model(df, model_path)
        df["qc_scorer"] = "model"
    else:
        df["qc_score"] = rule_score(df)
        df["qc_scorer"] = "rule"
    return df
