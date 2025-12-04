# pipeline.py
# DREAMT (64 Hz wearables) -> 30s epoch features -> nightly features -> trailing z -> HMM fit -> save model
import os, glob, math, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import medfilt
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump
from sklearn.metrics import silhouette_score

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data_raw" / "dreamt_64hz"   # folder with S###_whole_df.csv
FEAT_DIR = PROJECT_ROOT / "data_feat"
MODEL_DIR = PROJECT_ROOT / "models"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers (unchanged): rmssd_from_ibi, vec_mag  ----


# --- Helpers (add these) ---
def rmssd_from_ibi(ibi_ms: np.ndarray) -> float:
    """IBI in milliseconds -> RMSSD (ms). Returns NaN if not enough beats."""
    ibi_ms = np.asarray(ibi_ms, dtype=float)
    if ibi_ms.size < 3 or np.isnan(ibi_ms).sum() > ibi_ms.size - 3:
        return np.nan
    diff = np.diff(ibi_ms)
    return float(np.sqrt(np.nanmean(diff**2)))

def vec_mag(ax, ay, az):
    """Vector magnitude for 3-axis accelerometer arrays."""
    ax = np.asarray(ax, dtype=float)
    ay = np.asarray(ay, dtype=float)
    az = np.asarray(az, dtype=float)
    return np.sqrt(ax*ax + ay*ay + az*az)

# augmenting our data set with outputs from the hmm in order to stabilize prediction
# not in use decided not to overfit the model
def augment_from_hmm(df_feat, used_feats, hmm_like, n_factor=1.0, not_healthy_boost=1.5,
                     cov_shrink=0.7, clip=4.0, rng=0):
    import numpy as np, pandas as pd
    rs = np.random.RandomState(rng)
    means = hmm_like.means_
    covars = hmm_like.covars_
    K, d = means.shape
    # pick “not_healthy” by severity (+rhr +temp −rmssd −seff)
    weight_map = {"rhr_z": +1.0, "rmssd_z": -1.0, "temp_z": +1.0, "seff_z": -1.0}
    w = np.array([weight_map.get(f, 0.0) for f in used_feats], float)
    sev = means.dot(w)
    idx_not = int(np.argmax(sev))
    # allocation
    n_real = len(df_feat)
    n_syn = int(round(n_factor * n_real))
    if K == 2:
        p_not = not_healthy_boost / (1.0 + not_healthy_boost)
        alloc = {idx_not: int(round(n_syn * p_not)), 1 - idx_not: n_syn - int(round(n_syn * p_not))}
    else:
        per = n_syn // K
        alloc = {k: per for k in range(K)}
        alloc[0] += n_syn - per * K
    # draw
    syn_parts = []
    for k, n_k in alloc.items():
        if n_k <= 0: continue
        var = covars[k] * cov_shrink
        z = rs.normal(size=(n_k, d))
        x = means[k] + z * np.sqrt(var)
        x = np.clip(x, -clip, clip)
        syn_parts.append(pd.DataFrame(x, columns=used_feats))
    syn = pd.concat(syn_parts, ignore_index=True) if syn_parts else pd.DataFrame(columns=used_feats)
    syn["synthetic"] = True
    real = df_feat.copy()
    real["synthetic"] = False
    return pd.concat([real, syn], ignore_index=True)


# ---- NEW: normalize columns from various DREAMT/E4 dumps ----
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Map known names (case-insensitive)
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in {"timestamp","t_sec","t","time","seconds"}: rename_map[c] = "t_sec"
        elif cl in {"hr","heart_rate"}:                     rename_map[c] = "HR"
        elif cl in {"ibi_ms","ibi (ms)"}:                   rename_map[c] = "IBI_ms"
        elif cl in {"ibi","rr","rr_interval","ibi (s)"}:    rename_map[c] = "IBI"
        elif cl in {"temp","skin_temp","temperature"}:      rename_map[c] = "TEMP"
        elif cl in {"eda","gsr","electrodermal"}:           rename_map[c] = "EDA"
        elif cl in {"acc_x","x_acc","acc x"}:               rename_map[c] = "ACC_X"
        elif cl in {"acc_y","y_acc","acc y"}:               rename_map[c] = "ACC_Y"
        elif cl in {"acc_z","z_acc","acc z"}:               rename_map[c] = "ACC_Z"
        elif cl == "bvp":                                   rename_map[c] = "BVP"
    df = df.rename(columns=rename_map)

    if "t_sec" not in df.columns:
        raise ValueError("No time column found (TIMESTAMP/t/t_sec/time/seconds).")

    # Ensure numeric types
    for c in ["HR","TEMP","EDA","ACC_X","ACC_Y","ACC_Z","IBI","IBI_ms","BVP"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # IBI unit normalization: seconds -> ms if needed
    if "IBI_ms" not in df.columns and "IBI" in df.columns:
        med = np.nanmedian(df["IBI"])
        if np.isfinite(med):
            # If median < 10 it's almost surely seconds
            df["IBI_ms"] = df["IBI"] * (1000.0 if med < 10 else 1.0)
        else:
            df["IBI_ms"] = np.nan

    return df


def load_subject_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    keep = [c for c in ["t_sec","HR","IBI_ms","TEMP","EDA","ACC_X","ACC_Y","ACC_Z"] if c in df.columns]
    return df[keep].copy()


def resample_epochize(df, epoch_sec=30):
    # Requires 't_sec'; we just bin by floor(t/epoch)
    if "t_sec" not in df.columns:
        raise ValueError("t_sec missing after normalization.")
    df = df.copy()
    df["epoch_id"] = (df["t_sec"] // epoch_sec).astype(int)

    rows = []
    for eid, g in df.groupby("epoch_id"):
        row = {"epoch_id": eid, "t0_sec": eid * epoch_sec}
        row["hr_med"]     = np.nanmedian(g["HR"])   if "HR"   in g else np.nan
        row["rmssd"]      = rmssd_from_ibi(g["IBI_ms"].dropna().values) if "IBI_ms" in g and g["IBI_ms"].notna().sum() >= 3 else np.nan
        row["temp_mean"]  = np.nanmean(g["TEMP"])   if "TEMP" in g else np.nan
        row["eda_mean"]   = np.nanmean(g["EDA"])    if "EDA"  in g else np.nan
        row["eda_std"]    = np.nanstd(g["EDA"])     if "EDA"  in g else np.nan
        if set(["ACC_x","ACC_y","ACC_z"]).issubset(g.columns):
            vm = vec_mag(g["ACC_x"], g["ACC_y"], g["ACC_z"])
            row["acc_vm_mean"] = np.nanmean(vm)
            row["acc_vm_std"]  = np.nanstd(vm)
        else:
            row["acc_vm_mean"] = np.nan
            row["acc_vm_std"]  = np.nan
        rows.append(row)
    ep = pd.DataFrame(rows).sort_values("epoch_id").reset_index(drop=True)
    return ep

def nightly_aggregate(ep):
    if ep.empty:
        return None
    def p(q, x):
        return np.nanpercentile(x, q) if np.isfinite(x).sum() > 0 else np.nan

    feats = {}
    feats["rhr_night"]   = np.nanmedian(ep["hr_med"])
    feats["rmssd_night"] = np.nanmedian(ep["rmssd"])
    feats["temp_night"]  = np.nanmedian(ep["temp_mean"])
    # Sleep efficiency proxy from movement quietness
    if ep["acc_vm_mean"].notna().sum() > 10:
        vm = ep["acc_vm_mean"].values
        lo, hi = np.nanpercentile(vm, [10,90])
        rng = max(hi - lo, 1e-6)
        quiet = 1.0 - np.clip((vm - lo) / rng, 0, 1)
        feats["sleep_eff_proxy"] = float(np.nanmean(quiet))
    else:
        feats["sleep_eff_proxy"] = np.nan

    feats["hr_p90"]    = p(90, ep["hr_med"])
    feats["rmssd_p10"] = p(10, ep["rmssd"])
    feats["temp_p90"]  = p(90, ep["temp_mean"])
    feats["eda_mean"]  = np.nanmean(ep["eda_mean"]) if "eda_mean" in ep else np.nan
    feats["eda_std"]   = np.nanstd(ep["eda_mean"])  if "eda_mean" in ep else np.nan

    night = pd.DataFrame([feats])
    # Require at least HR or TEMP to exist; otherwise mark unusable
    if np.isnan(night[["rhr_night","temp_night"]].values).all():
        return None
    return night

def build_dreamt_features():
    files = sorted(DATA_RAW.glob("S*_whole_df.csv")) or sorted(DATA_RAW.glob("*.csv"))
    print(f"[INFO] Looking in {DATA_RAW}; found {len(files)} CSV(s)")
    nights = []
    bad = []

    for fpath in files:
        try:
            subj_id = fpath.stem.split("_")[0]  # "S002"
            raw = load_subject_csv(fpath)
            present = ", ".join([c for c in ["HR","IBI_ms","TEMP","EDA","ACC_X","ACC_Y","ACC_Z"] if c in raw.columns])
            print(f"  - {fpath.name}: cols[{present}] rows={len(raw)}")

            # Split into 24h bins per subject
            raw = raw.copy()
            raw["subject"] = subj_id
            raw["day_id"] = (raw["t_sec"] // 86400).astype(int)

            for (sid, day), g in raw.groupby(["subject", "day_id"], sort=True):
                ep = resample_epochize(g, epoch_sec=30)
                night = nightly_aggregate(ep)
                if night is None:
                    continue
                night["subject"] = sid
                # Turn day_id (0,1,2,...) into a real date (any origin works; we just need ordering)
                night["date"] = pd.to_datetime(day, unit="D", origin="1970-01-01")
                nights.append(night)

        except Exception as e:
            bad.append((fpath.name, str(e)))

    if not nights:
        print("[ERROR] No usable nights were extracted.")
        if bad:
            print("Details:")
            for name, msg in bad[:20]:
                print(f"   * {name}: {msg}")
        raise RuntimeError(f"No DREAMT features extracted from {DATA_RAW}. Check columns and file format.")

    df = pd.concat(nights, ignore_index=True)
    df = df.sort_values(["subject","date"]).reset_index(drop=True)

    # ---- Subject-wise trailing z-scores (short window) + cohort fallback ----
    def trailing_z(s, win=14, minp=3, shift1=True):
        m  = s.rolling(win, min_periods=minp).mean()
        sd = s.rolling(win, min_periods=minp).std(ddof=0)
        if shift1:
            m = m.shift(1); sd = sd.shift(1)
        z = (s - m) / (sd.replace(0, np.nan))
        return z

    def robust_z(x: pd.Series):
        x = pd.to_numeric(x, errors="coerce")
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if not np.isfinite(mad) or mad == 0:
            # fall back to std if MAD is degenerate
            sd = np.nanstd(x, ddof=0)
            return (x - med) / (sd if sd not in (0, np.nan) else 1.0)
        return (x - med) / (1.4826 * mad)

    # Cohort robust z (used as fallback for single-night subjects)
    df["rhr_z_cohort"]   = robust_z(df["rhr_night"])
    df["rmssd_z_cohort"] = -robust_z(df["rmssd_night"])  # lower RMSSD = worse → negative sign
    df["temp_z_cohort"]  = robust_z(df["temp_night"])
    df["seff_z_cohort"]  = -robust_z(df["sleep_eff_proxy"])

    def add_z(g):
        out = g.copy()
        # rolling per-subject (shifted so tonight not in baseline)
        out["rhr_z"]   = trailing_z(out["rhr_night"])
        out["rmssd_z"] = trailing_z(out["rmssd_night"]) * (-1.0)
        out["temp_z"]  = trailing_z(out["temp_night"])
        out["seff_z"]  = trailing_z(out["sleep_eff_proxy"]) * (-1.0)
        return out

    # Use include_groups=False to silence the deprecation warning
    df = df.groupby("subject", group_keys=False).apply(add_z)

    # Fill any NaNs from the cohort robust z (handles single-night subjects)
    for col, cfb in [("rhr_z","rhr_z_cohort"),
                    ("rmssd_z","rmssd_z_cohort"),
                    ("temp_z","temp_z_cohort"),
                    ("seff_z","seff_z_cohort")]:
        df[col] = df[col].fillna(df[cfb])

    # Clean up
    df = df.drop(columns=["rhr_z_cohort","rmssd_z_cohort","temp_z_cohort","seff_z_cohort"]).reset_index(drop=True)

    # Keep nights where at least ONE z-feature exists
    keep = df[["rhr_z","rmssd_z","temp_z","seff_z"]].notna().any(axis=1)
    df = df[keep].reset_index(drop=True)


    out_path = FEAT_DIR / "dreamt_nightly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] DREAMT nightly features -> {out_path} (rows={len(df)})")
    if len(df) == 0:
        raise RuntimeError("Feature table empty after z-scoring.")
    return df

def train_val_split_by_subject(df, val_frac=0.2, seed=0):
    subjects = df["subject"].dropna().unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n_val = max(1, int(round(len(subjects) * val_frac)))
    val_subj = list(subjects[:n_val])
    train_subj = [s for s in subjects if s not in val_subj]

    df_train = df[df["subject"].isin(train_subj)].reset_index(drop=True)
    df_val   = df[df["subject"].isin(val_subj)].reset_index(drop=True)

    print(f"[SPLIT] train_subj={len(train_subj)} val_subj={len(val_subj)}")
    return df_train, df_val, train_subj, val_subj

def fit_hmm_on_features(df, K=3, seed=0):
    # Prefer physiological trio; fall back to seff if coverage is high
    feat_candidates = ["rhr_z", "rmssd_z", "temp_z", "seff_z"]
    coverage = df[feat_candidates].notna().mean()
    used_feats = [c for c in ["rhr_z", "rmssd_z", "temp_z"] if coverage.get(c, 0) >= 0.40]
    if len(used_feats) < 2:
        # add seff_z if needed
        if coverage.get("seff_z", 0) >= 0.60:
            used_feats += ["seff_z"]
    if len(used_feats) < 2:
        raise RuntimeError(f"Not enough usable features. Coverage={coverage.to_dict()}")

    # For now, keep trio unless seff_z was explicitly added above
    if "seff_z" not in used_feats:
        used_feats = ["rhr_z", "rmssd_z", "temp_z"]

    X_raw = df[used_feats].astype("float64").values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    # Independent nights: one-step sequences
    lengths = [1] * len(X)

    # Init means with KMeans
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(X)
    means0 = km.cluster_centers_

    # Fixed, sticky transitions (healthy stays healthy)
    T = np.array([[0.96, 0.04],
                  [0.10, 0.90]])

    hmm = GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=400,
        random_state=seed,
        init_params="sc",   # do NOT include 't' or 'm' so they aren’t overwritten
        params="smc"
    )
    hmm.means_ = means0
    hmm.transmat_ = T
    hmm.min_covar = 1e-3   # small regularization
    hmm.startprob_ = np.array([0.90, 0.10], float)

    hmm.fit(X, lengths=lengths)

    # Label mapping: +rhr +temp −rmssd → “not_healthy”
    w = np.array([+1.0, -1.0, +1.0])[:len(used_feats)]
    sev = hmm.means_.dot(w)   # higher = worse
    order = np.argsort(sev)
    labels = np.empty(K, dtype=object)
    labels[order[0]] = "healthy"
    labels[order[1]] = "not_healthy"

    print(f"[OK] HMM fitted. Using features: {used_feats}")
    return hmm, labels, used_feats, imputer



if __name__ == "__main__":
    df_all = build_dreamt_features()

    df_train, df_val, train_subj, val_subj = train_val_split_by_subject(df_all, val_frac=0.20, seed=0)

    hmm, labels, used_feats, imputer = fit_hmm_on_features(df_train, K=2, seed=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1) save model
    dump(hmm, MODEL_DIR / "dreamt_hmm.joblib")
    dump(imputer, MODEL_DIR / "dreamt_imputer.joblib")

    # 2) save split info
    with open(MODEL_DIR / "train_val_split.json", "w") as f:
        json.dump(
            {
                "train_subjects": train_subj,
                "val_subjects": val_subj,
                "val_frac": 0.20,
                "seed": 0,
            },
            f,
            indent=2,
        )

    # 3) save HMM metadata
    meta = {
        "used_feats": used_feats,
        "labels": {int(i): str(l) for i, l in enumerate(labels)},
    }
    with open(MODEL_DIR / "hmm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] model + split + meta saved.")


