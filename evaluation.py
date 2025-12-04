# evaluation.py
# evaluation.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import silhouette_score

from pipeline import FEAT_DIR, MODEL_DIR  # uses your existing constants


# ---------- Core evaluation ----------

def evaluate_hmm(df_train, df_val, hmm, used_feats, labels, imputer):
    """
    Compute a bundle of evaluation metrics for the HMM on train/val splits.

    Returns:
        metrics: dict (JSON-serializable) with:
          - counts and split info
          - log-likelihoods (total & per-sample)
          - posterior entropy
          - state proportions
          - state means / covars
          - silhouette score on train (if possible)
    """
    metrics = {}

    # --- basic split info ---
    metrics["n_subjects_train"] = int(df_train["subject"].nunique())
    metrics["n_subjects_val"] = int(df_val["subject"].nunique())
    metrics["n_nights_train"] = int(len(df_train))
    metrics["n_nights_val"] = int(len(df_val))
    metrics["used_features"] = list(used_feats)
    metrics["state_labels"] = {int(k): str(v) for k, v in enumerate(labels)}

    # --- build feature matrices using the same imputer ---
    Xtr_raw = df_train[used_feats].astype("float64").values
    Xval_raw = df_val[used_feats].astype("float64").values

    Xtr = imputer.transform(Xtr_raw)
    Xval = imputer.transform(Xval_raw)

    # --- log-likelihoods ---
    loglike_train = hmm.score(Xtr)
    loglike_val = hmm.score(Xval)

    metrics["loglike_train_total"] = float(loglike_train)
    metrics["loglike_val_total"] = float(loglike_val)
    metrics["loglike_train_avg_per_sample"] = float(
        loglike_train / max(len(Xtr), 1)
    )
    metrics["loglike_val_avg_per_sample"] = float(
        loglike_val / max(len(Xval), 1)
    )

    # --- posteriors ---
    post_tr = hmm.predict_proba(Xtr)
    post_val = hmm.predict_proba(Xval)

    def avg_entropy(P):
        eps = 1e-12
        H = -np.sum(P * np.log(P + eps), axis=1)
        return float(np.mean(H)) if len(H) > 0 else float("nan")

    metrics["posterior_entropy_train"] = avg_entropy(post_tr)
    metrics["posterior_entropy_val"] = avg_entropy(post_val)

    # --- state proportions (expected frequency of each state) ---
    metrics["state_proportions_train"] = {
        str(labels[k]): float(post_tr[:, k].mean())
        for k in range(hmm.n_components)
    }
    metrics["state_proportions_val"] = {
        str(labels[k]): float(post_val[:, k].mean())
        for k in range(hmm.n_components)
    }

    # --- hard assignments and silhouette on train ---
    try:
        hard_states = np.argmax(post_tr, axis=1)
        if len(np.unique(hard_states)) > 1 and len(hard_states) > hmm.n_components:
            sil = silhouette_score(Xtr, hard_states)
            metrics["silhouette_train"] = float(sil)
        else:
            metrics["silhouette_train"] = None
    except Exception as e:
        metrics["silhouette_train"] = None
        metrics["silhouette_error"] = str(e)

    # --- state means / covars mapped to labels ---
    means = np.array(hmm.means_)
    covars = np.array(hmm.covars_)
    cov_type = getattr(hmm, "covariance_type", "diag")

    means_dict = {}
    covars_dict = {}

    for k in range(hmm.n_components):
        label = str(labels[k])

        # Means: always (n_components, n_features)
        means_dict[label] = {
            feat: float(means[k, i]) for i, feat in enumerate(used_feats)
        }

        # Covariances: handle whatever shape we get
        cov_k = np.array(covars[k])

        if cov_type == "diag":
            # Could be (n_features,) or (1, n_features) etc.
            diag = np.atleast_1d(cov_k).ravel()
            covars_dict[label] = {
                feat: float(diag[i]) for i, feat in enumerate(used_feats)
            }
        else:
            # Treat as full matrix; at least 2D
            cov_mat = np.atleast_2d(cov_k)
            diag = np.diag(cov_mat)
            covars_dict[label] = {
                "full": cov_mat.tolist(),
                "diag": {
                    feat: float(diag[i]) for i, feat in enumerate(used_feats)
                },
            }

    metrics["state_means"] = means_dict
    metrics["state_covars"] = covars_dict
#
    


    # --- transitions & start probabilities ---
    metrics["transmat"] = hmm.transmat_.tolist()
    metrics["startprob"] = hmm.startprob_.tolist()

    return metrics


# ---------- Helpers for posteriors & plots ----------

def compute_posteriors(df, hmm, used_feats, imputer):
    """Return (X, posteriors, hard_states) for a dataframe."""
    X_raw = df[used_feats].astype("float64").values
    X = imputer.transform(X_raw)
    post = hmm.predict_proba(X)
    hard = np.argmax(post, axis=1)
    return X, post, hard


def plot_p_not_histogram(post_tr, post_val, labels, out_path):
    """Histogram of p(not_healthy) for train and val."""
    # find not_healthy index if present, else fallback to 1
    lbl_list = list(labels)
    idx_not = lbl_list.index("not_healthy") if "not_healthy" in lbl_list else 1

    p_not_tr = post_tr[:, idx_not]
    p_not_val = post_val[:, idx_not]

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 30)
    plt.hist(p_not_tr, bins=bins, alpha=0.6, label="train", density=True)
    plt.hist(p_not_val, bins=bins, alpha=0.6, label="val", density=True)
    plt.xlabel("Posterior p(not_healthy)")
    plt.ylabel("Density")
    plt.title("Distribution of p(not_healthy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_state_scatter(df_train, Xtr, hard_states, used_feats, labels, out_path):
    """
    Scatter of two main features colored by hard state.
    Picks first two features in used_feats.
    """
    if len(used_feats) < 2 or Xtr.shape[1] < 2:
        return  # not enough features to plot

    f1, f2 = used_feats[0], used_feats[1]
    x = Xtr[:, 0]
    y = Xtr[:, 1]

    plt.figure(figsize=(6, 5))
    for k in range(len(labels)):
        mask = hard_states == k
        if not np.any(mask):
            continue
        plt.scatter(
            x[mask],
            y[mask],
            s=10,
            alpha=0.6,
            label=str(labels[k]),
        )
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title("Train nights in feature space by HMM state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_state_means_bar(hmm, used_feats, labels, out_path):
    """
    Bar chart of state means across features.
    x-axis: features, grouped by state.
    """
    means = hmm.means_
    n_states, n_feats = means.shape

    x = np.arange(n_feats)
    width = 0.8 / max(n_states, 1)

    plt.figure(figsize=(7, 4))
    for k in range(n_states):
        offs = (k - (n_states - 1) / 2.0) * width
        plt.bar(
            x + offs,
            means[k, :],
            width=width,
            alpha=0.8,
            label=str(labels[k]),
        )
    plt.xticks(x, used_feats, rotation=30, ha="right")
    plt.ylabel("State mean (z-score space)")
    plt.title("HMM state means by feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Main entrypoint ----------

def main():
    # 1) Load nightly features
    feat_path = FEAT_DIR / "dreamt_nightly.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    df_all = pd.read_parquet(feat_path)

    # 2) Load train/val split (subject-wise) saved by pipeline.py
    split_path = MODEL_DIR / "train_val_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}. "
            "Run pipeline.py to generate train/val split."
        )
    with open(split_path, "r") as f:
        split = json.load(f)
    train_subj = set(split["train_subjects"])
    val_subj = set(split["val_subjects"])

    df_train = df_all[df_all["subject"].isin(train_subj)].reset_index(drop=True)
    df_val = df_all[df_all["subject"].isin(val_subj)].reset_index(drop=True)

    # 3) Load model + imputer + meta
    hmm = load(MODEL_DIR / "dreamt_hmm.joblib")
    imputer = load(MODEL_DIR / "dreamt_imputer.joblib")

    meta_path = MODEL_DIR / "hmm_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Meta file not found: {meta_path}. "
            "Ensure pipeline.py saved hmm_meta.json."
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)

    used_feats = meta["used_feats"]
    # reconstruct labels list in order [0..n_components-1]
    labels = [meta["labels"][str(i)] for i in range(hmm.n_components)]

    # 4) Compute metrics
    metrics = evaluate_hmm(df_train, df_val, hmm, used_feats, labels, imputer)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    eval_json = MODEL_DIR / "dreamt_hmm_eval.json"
    with open(eval_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Evaluation metrics saved to {eval_json}")

    # 5) Compute posteriors again for plotting
    Xtr, post_tr, hard_tr = compute_posteriors(df_train, hmm, used_feats, imputer)
    Xval, post_val, _ = compute_posteriors(df_val, hmm, used_feats, imputer)

    # 6) Plots
    plot_p_not_histogram(
        post_tr,
        post_val,
        labels,
        MODEL_DIR / "dreamt_p_not_hist.png",
    )
    plot_state_scatter(
        df_train,
        Xtr,
        hard_tr,
        used_feats,
        labels,
        MODEL_DIR / "dreamt_state_scatter_train.png",
    )
    plot_state_means_bar(
        hmm,
        used_feats,
        labels,
        MODEL_DIR / "dreamt_state_means.png",
    )

    print("[OK] Plots saved to:")
    print("   -", MODEL_DIR / "dreamt_p_not_hist.png")
    print("   -", MODEL_DIR / "dreamt_state_scatter_train.png")
    print("   -", MODEL_DIR / "dreamt_state_means.png")


if __name__ == "__main__":
    main()

