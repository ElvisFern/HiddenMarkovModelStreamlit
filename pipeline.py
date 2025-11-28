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
"""
builds feature per person instead of per night
not good as it does not make efficient use of our data points

def build_dreamt_features():
    files = sorted(DATA_RAW.glob("S*_whole_df.csv"))
    if not files:
        files = sorted(DATA_RAW.glob("*.csv"))

    print(f"[INFO] Looking in {DATA_RAW}; found {len(files)} CSV(s)")
    records = []
    bad = []

    for fpath in files:
        try:
            subj_id = fpath.stem.split("_")[0]  # "S002"
            raw = load_subject_csv(fpath)
            # quick column report
            present = ", ".join([c for c in ["HR","IBI_ms","TEMP","EDA","ACC_x","ACC_y","ACC_z"] if c in raw.columns])
            print(f"  - {fpath.name}: cols[{present}] rows={len(raw)}")
            ep  = resample_epochize(raw, epoch_sec=30)
            night = nightly_aggregate(ep)
            if night is None:
                bad.append((fpath.name, "no usable epoch features"))
                continue
            night["subject"] = subj_id
            night["date"] = pd.to_datetime("2020-01-01")  # placeholder
            records.append(night)
        except Exception as e:
            bad.append((fpath.name, str(e)))

    if not records:
        print("[ERROR] No usable nights were extracted.")
        if bad:
            print("Details:")
            for name, msg in bad[:20]:
                print(f"   * {name}: {msg}")
        raise RuntimeError(f"No DREAMT features extracted from {DATA_RAW}. Check columns and file format.")

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values(["subject","date"]).reset_index(drop=True)

    # Cohort (global) z-scores so single-night subjects aren't dropped
    def zscore(x):
        m = np.nanmean(x); s = np.nanstd(x, ddof=0)
        return (x - m) / (s if s > 1e-8 else 1.0)

    df["rhr_z"]   = zscore(df["rhr_night"])
    df["rmssd_z"] = zscore(df["rmssd_night"]) * (-1.0)
    df["temp_z"]  = zscore(df["temp_night"])
    df["seff_z"]  = zscore(df["sleep_eff_proxy"]) * (-1.0)

    # Keep rows with at least two valid features (looser filter)
    keep_mask = (
        df[["rhr_z","rmssd_z","temp_z","seff_z"]].isna().sum(axis=1) <= 2
    )
    df = df[keep_mask].reset_index(drop=True)

    out_path = FEAT_DIR / "dreamt_nightly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[OK] DREAMT nightly features -> {out_path} (rows={len(df)})")

    if len(df) == 0:
        raise RuntimeError("Feature table is empty after z-scoring. Check which signals are present and relax filters.")

    return df
"""
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

"""
fits on old feature set
simplifying the model 

def fit_hmm_on_features(df, K=4, seed=0):
    # Candidate features in priority order
    feat_candidates = ["rhr_z", "rmssd_z", "temp_z", "seff_z"]

    # Keep features with at least 60% non-NaN coverage
    coverage = df[feat_candidates].notna().mean()
    used_feats = [c for c in feat_candidates if coverage.get(c, 0.0) >= 0.60]
    if len(used_feats) < 2:
        raise RuntimeError(f"Not enough usable features. Coverage={coverage.to_dict()}")

    X_raw = df[used_feats].astype(np.float64).values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)  # no NaNs from here on

    # KMeans init
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(X)
    hmm = GaussianHMM(n_components=K, covariance_type="full", n_iter=300, random_state=seed)
    hmm.means_ = km.cluster_centers_

    trans = np.full((K, K), 1.0 / K)
    np.fill_diagonal(trans, 0.70)
    for i in range(K):
        trans[i] /= trans[i].sum()
    hmm.transmat_ = trans
    hmm.fit(X)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(hmm, MODEL_DIR / "model.pkl")
    dump(imputer, MODEL_DIR / "imputer.pkl")
    # Save feature list so validation/app use the same columns
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump({"features": used_feats}, f)

    # Label mapping by severity (sum of means)
    means = hmm.means_
    severity = means.sum(axis=1)
    idx = np.argsort(severity)  # low -> high
    labels = np.empty(hmm.n_components, dtype=object)
    if hmm.n_components == 3:
        labels[idx[0]] = "healthy"; labels[idx[1]] = "recovery"; labels[idx[2]] = "symptomatic"
    else:
        labels[idx[0]] = "healthy"; labels[idx[1]] = "prodrome"; labels[idx[2]] = "recovery"; labels[idx[3]] = "symptomatic"
    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump({str(i): labels[i] for i in range(hmm.n_components)}, f)

    print(f"[OK] HMM saved. Using features: {used_feats}")
    return hmm, labels
"""

def fit_hmm_on_features(df, K=3, seed=0):
    # Prefer physiological trio; fall back to seff if coverage is high
    feat_candidates = ["rhr_z", "rmssd_z", "temp_z", "seff_z"]
    coverage = df[feat_candidates].notna().mean()
    used_feats = [c for c in ["rhr_z","rmssd_z","temp_z"] if coverage.get(c,0) >= 0.40]
    if len(used_feats) < 2:
        # add seff_z if needed
        if coverage.get("seff_z",0) >= 0.60:
            used_feats += ["seff_z"]
    if len(used_feats) < 2:
        raise RuntimeError(f"Not enough usable features. Coverage={coverage.to_dict()}")

    X_raw = df[used_feats].astype(np.float64).values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    # sequence lengths per subject (nights per subject)
    lengths = df.groupby("subject")["date"].count().tolist()

    # KMeans init + sticky-ish transitions
    km = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(X)
    trans = np.full((K, K), 1.0 / K)
    np.fill_diagonal(trans, 0.88)  # a bit sticky
    for i in range(K): trans[i] /= trans[i].sum()

    hmm = GaussianHMM(
        n_components=K,
        covariance_type="diag",  # tighter clusters
        n_iter=400,
        random_state=seed,
        init_params="sc"         # keep our means_ and transmat_
    )
    hmm.means_ = km.cluster_centers_
    hmm.transmat_ = trans

    hmm.fit(X, lengths=lengths)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(hmm, MODEL_DIR / "model.pkl")
    dump(imputer, MODEL_DIR / "imputer.pkl")
    with open(MODEL_DIR / "features.json", "w") as f:
        json.dump({"features": used_feats}, f)

    # Data-driven state labels (severity = +rhr +temp -rmssd)
    w = np.array([+1.0, -1.0, +1.0, 0.0])[:len(used_feats)]
    sev = hmm.means_.dot(w)
    order = np.argsort(sev)  # low -> high
    labels = np.empty(K, dtype=object)
    if K == 3:
        labels[order[0]]="healthy"; labels[order[1]]="recovery"; labels[order[2]]="symptomatic"
    else:
        labels[order[0]]="healthy"; labels[order[1]]="prodrome"; labels[order[2]]="recovery"; labels[order[3]]="symptomatic"
    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump({str(i): labels[i] for i in range(K)}, f)

    print(f"[OK] HMM saved. Using features: {used_feats}")
    return hmm, labels


if __name__ == "__main__":
    df = build_dreamt_features()
    fit_hmm_on_features(df, K=3, seed=0)

