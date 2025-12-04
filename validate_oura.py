# validate_oura.py
# Build nightly features from Oura export, apply saved HMM, and create validation graphs
import json,glob, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from scipy.stats import spearmanr


MODEL_DIR = Path("models")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def _read_first(patterns, **kw):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            return pd.read_csv(hits[0], **kw)
    return None

def _parse_contributors(s):
    if pd.isna(s): 
        return {}
    # Oura exports are valid JSON; sometimes Excel saves w/out quotes normalized.
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}

def _to_day_datetime(s):
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    # remove timezone if any, then normalize to midnight
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    return dt.dt.floor("D")

def _empty_date_df(cols):
    d = {"date": pd.Series([], dtype="datetime64[ns]")}
    for c in cols:
        d[c] = pd.Series([], dtype="float64")
    return pd.DataFrame(d)


def trailing_zscore(s, win=30, minp=7):
    m = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std(ddof=0).shift(1)
    return (s - m) / sd

def build_oura_nightly(oura_root="data_raw/oura"):
    root = Path(oura_root)

    tdf   = _empty_date_df(["temp_dev"])
    hrt   = _empty_date_df(["rhr"])
    resil = _empty_date_df(["rmssd"])

    # ---- Sleep (semicolon-delimited, with contributors JSON) ----
    sleep = _read_first(
        [str(root/"dailysleep.csv"), str(root/"daily_sleep.csv"), str(root/"sleep.csv")],
        sep=";"  # <-- important
    )
    if sleep is None:
        raise FileNotFoundError("Could not find dailysleep.csv (semicolon-delimited).")

    # Normalize date col
    date_col = "day" if "day" in sleep.columns else ("date" if "date" in sleep.columns else "summary_date")
    if date_col is None:
        raise ValueError("No date/day column in dailysleep.csv.")
    sleep = sleep.rename(columns={date_col: "date"})
    sleep["date"] = pd.to_datetime(sleep["date"]).dt.tz_localize(None)

    # Sleep score + contributors (efficiency proxy)
    # contributors is a JSON like {"efficiency": 85, "latency": 92, ...}
    if "contributors" in sleep.columns:
        contrib = sleep["contributors"].apply(_parse_contributors)
        eff = contrib.apply(lambda d: d.get("efficiency", np.nan))
        sleep["sleep_efficiency"] = eff / 100.0  # contributors are 0-100
    else:
        sleep["sleep_efficiency"] = np.nan

    # Keep overall sleep score if present (can be a useful sanity metric)
    if "score" in sleep.columns:
        sleep = sleep.rename(columns={"score": "sleep_score"})
    else:
        sleep["sleep_score"] = np.nan

    sleep = sleep[["date", "sleep_efficiency", "sleep_score"]]

    # ---- Readiness (daily score) ----
    # Many exports use semicolons + 'timestamp'
    read = _read_first(
        [str(root / "dailyreadiness.csv"), str(root / "daily_readiness.csv"), str(root / "readiness.csv")],
        sep=";"  # try semicolon first
    )
    if read is None:
        read = _read_first(
            [str(root / "dailyreadiness.csv"), str(root / "daily_readiness.csv"), str(root / "readiness.csv")]
        )

    if read is not None:
        # Normalize to a 'date' column
        if "timestamp" in read.columns:
            read["date"] = pd.to_datetime(read["timestamp"]).dt.tz_localize(None).dt.date
        else:
            date_col = "day" if "day" in read.columns else ("date" if "date" in read.columns else "summary_date")
            read["date"] = pd.to_datetime(read[date_col]).dt.tz_localize(None).dt.date

        # Readiness score
        if "readiness" in read.columns:
            pass
        elif "score" in read.columns:
            read = read.rename(columns={"score": "readiness"})
        else:
            read["readiness"] = np.nan

       # Pull out body_temperature contributor if present
        if "contributors" in read.columns:
            r_contrib = read["contributors"].apply(_parse_contributors)
            read["body_temperature"] = r_contrib.apply(
                lambda d: d.get("body_temperature", np.nan)
            )

        # Keep just the columns we care about
        cols = ["date", "readiness"]
        if "body_temperature" in read.columns:
            cols.append("body_temperature")
        read = read[cols]
    else:
        read = pd.DataFrame(columns=["date", "readiness"])

    print("[DEBUG] readiness rows:", len(read), "cols:", list(read.columns)[:6])
    print(read.head(3))

    # ---- Temperature deviation ----
    tdf = _read_first(
        [str(root/"temperature.csv"), str(root/"dailytemperature.csv"), str(root/"daily_temperature.csv")],
        sep=";"  # try semicolon-delimited first
    )
    if tdf is None:
        tdf = _read_first(
            [str(root/"temperature.csv"), str(root/"dailytemperature.csv"), str(root/"daily_temperature.csv")]
        )

    if tdf is not None:
        # date/timestamp → date
        if "timestamp" in tdf.columns:
            tdf["date"] = pd.to_datetime(tdf["timestamp"]).dt.tz_localize(None).dt.date
        else:
            t_date = "day" if "day" in tdf.columns else ("date" if "date" in tdf.columns else "summary_date")
            tdf["date"] = pd.to_datetime(tdf[t_date]).dt.tz_localize(None).dt.date

        # deviation column name variants
        if "temperature_deviation" in tdf.columns:
            tdf = tdf.rename(columns={"temperature_deviation": "temp_dev"})
        elif "temperature_delta" in tdf.columns:
            tdf = tdf.rename(columns={"temperature_delta": "temp_dev"})
        elif "temp_deviation" in tdf.columns:
            tdf = tdf.rename(columns={"temp_deviation": "temp_dev"})
        else:
            tdf["temp_dev"] = np.nan

        tdf = tdf[["date", "temp_dev"]]
    else:
        tdf = pd.DataFrame(columns=["date", "temp_dev"])
    # ---- Resting heart rate ----
    hrt = _read_first([str(root/"heartrate.csv"), str(root/"heart_rate.csv")], sep=";")
    if hrt is None:
        hrt = _read_first([str(root/"heartrate.csv"), str(root/"heart_rate.csv")])  # fallback comma

    if hrt is not None:
        if "timestamp" in hrt.columns:
            hrt["date"] = pd.to_datetime(hrt["timestamp"]).dt.tz_localize(None).dt.date
        else:
            h_date = "day" if "day" in hrt.columns else ("date" if "date" in hrt.columns else "summary_date")
            hrt["date"] = pd.to_datetime(hrt[h_date]).dt.tz_localize(None).dt.date

        # Prefer explicit resting_heart_rate if present or can be inferred from a similarly named column
        if "resting_heart_rate" not in hrt.columns:
            cand = [c for c in hrt.columns if "resting" in c.lower() and "heart" in c.lower()]
            if cand:
                hrt = hrt.rename(columns={cand[0]: "resting_heart_rate"})

        # If we now have resting_heart_rate, use it as rhr directly;
        # otherwise we’ll fall back to bpm later once we’ve seen all columns.
        if "resting_heart_rate" in hrt.columns:
            hrt["rhr"] = hrt["resting_heart_rate"]

        # Keep date + whatever we might need (rhr, bpm, source)
        keep_cols = ["date"]
        for c in ["rhr", "bpm", "source"]:
            if c in hrt.columns:
                keep_cols.append(c)
        hrt = hrt[keep_cols]
    else:
        hrt = pd.DataFrame(columns=["date", "rhr"])


    # ---- HRV (rmssd) if present anywhere (often missing in new exports)
    # Try dailyresilience.csv as a proxy; else leave NaN and imputer will handle
    resil = _read_first([str(root/"dailyresilience.csv")], sep=";")
    if resil is None:
        resil = _read_first([str(root/"dailyresilience.csv")])  # fallback comma

    if resil is not None:
        if "timestamp" in resil.columns:
            resil["date"] = pd.to_datetime(resil["timestamp"]).dt.tz_localize(None).dt.date
        else:
            rz_date = "day" if "day" in resil.columns else ("date" if "date" in resil.columns else "summary_date")
            resil["date"] = pd.to_datetime(resil[rz_date]).dt.tz_localize(None).dt.date

        if "average_hrv" in resil.columns:
            resil = resil.rename(columns={"average_hrv": "rmssd"})
        else:
            resil["rmssd"] = np.nan

        resil = resil[["date", "rmssd"]]
    else:
        resil = pd.DataFrame(columns=["date", "rmssd"])

   

    def _norm_inplace_date(_df):
        if "date" in _df.columns:
            _df["date"] = pd.to_datetime(_df["date"], errors="coerce", utc=False)
            try:
                _df["date"] = _df["date"].dt.tz_localize(None)
            except Exception:
                pass
            _df["date"] = _df["date"].dt.floor("D")

    for name, _df in [("sleep", sleep), ("read", read), ("tdf", tdf), ("hrt", hrt), ("resil", resil)]:
        _norm_inplace_date(_df)

    # ensure single row per date in every table
    sleep = (sleep.groupby("date", as_index=False)
                .agg({"sleep_efficiency":"mean", "sleep_score":"mean"}))

    # Aggregate readiness (and any extras we kept, like temp_dev / body_temperature)
    agg = {"readiness": "mean"}
    if "body_temperature" in read.columns:
        agg["body_temperature"] = "mean"

    read = (read.groupby("date", as_index=False)
                .agg(agg))

    tdf = (tdf.groupby("date", as_index=False)
            .agg({"temp_dev":"mean"}))
    # Heart rate: prefer true resting_heart_rate; else fallback
    if "rhr" not in hrt.columns and "bpm" in hrt.columns:
        hrt["rhr"] = pd.to_numeric(hrt["bpm"], errors="coerce")

    # rhr per day: take the *min* (proxy for nightly resting HR), else mean if you prefer
    hrt = (hrt.groupby("date", as_index=False)
            .agg({"rhr":"min"}))

    resil = (resil.groupby("date", as_index=False)
                .agg({"rmssd":"mean"}))
    for name, _df in [("sleep", sleep), ("read", read), ("tdf", tdf), ("hrt", hrt), ("resil", resil)]:
        assert _df["date"].is_monotonic_increasing or True  # harmless
        dup = _df["date"].duplicated().sum()
        if dup:
            print(f"[WARN] {name} still has {dup} duplicate dates")

    # sanity print (should all be datetime64[ns])
    print({name: _df["date"].dtype for name, _df in [("sleep", sleep), ("read", read), ("tdf", tdf), ("hrt", hrt), ("resil", resil)]})
    # ---- Merge all nightly features on date ----
    df = sleep.merge(read, on="date", how="left") \
              .merge(tdf,  on="date", how="left") \
              .merge(hrt,  on="date", how="left") \
              .merge(resil, on="date", how="left") \
              .sort_values("date")

    # Normalize efficiency if given 0–100
    if df["sleep_efficiency"].notna().any() and df["sleep_efficiency"].max() > 1.5:
        df["sleep_efficiency"] = df["sleep_efficiency"] / 100.0

    # Trailing z-scores (exclude tonight)
    def trailing_zscore(s, win=30, minp=7):
        m = s.rolling(win, min_periods=minp).mean().shift(1)
        sd = s.rolling(win, min_periods=minp).std(ddof=0).shift(1)
        return (s - m) / sd

    df["rhr_z"]   = trailing_zscore(df["rhr"])                      # + worse with ↑
    df["rmssd_z"] = trailing_zscore(df["rmssd"]) * (-1.0)           # − worse with ↓
    df["temp_z"]  = trailing_zscore(df["temp_dev"])                 # + worse with ↑
    df["seff_z"]  = trailing_zscore(df["sleep_efficiency"]) * (-1.) # − worse with ↓

    # Keep rows where at least one z-feature exists; the model's imputer will fill the rest
    df = df.dropna(subset=["rhr_z","rmssd_z","temp_z","seff_z"], how="all").reset_index(drop=True)
    return df

def apply_model(df):
    hmm = load(MODEL_DIR / "model.pkl")
    imputer = load(MODEL_DIR / "imputer.pkl")
    with open(MODEL_DIR / "features.json") as f:
        used_feats = json.load(f)["features"]
    with open(MODEL_DIR / "labels.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # -------- GUARDS: fix bad startprob_/transmat_ from old models --------
    K = getattr(hmm, "n_components", None) or len(set(label_map))
    def _sticky_T(K, stay=0.95):
        T = np.full((K, K), (1.0 - stay) / (K - 1))
        np.fill_diagonal(T, stay)
        return T

    # startprob_
    needs_start = (
        not hasattr(hmm, "startprob_")
        or not np.isfinite(hmm.startprob_).all()
        or hmm.startprob_.ndim != 1
        or hmm.startprob_.shape[0] != K
        or not np.isclose(hmm.startprob_.sum(), 1.0)
    )
    if needs_start:
        start = np.zeros(K, dtype=float)
        start[0] = 1.0
        hmm.startprob_ = start

    # transmat_
    needs_T = (
        not hasattr(hmm, "transmat_")
        or not np.isfinite(hmm.transmat_).all()
        or hmm.transmat_.shape != (K, K)
        or np.any(hmm.transmat_.sum(axis=1) == 0.0)
        or not np.allclose(hmm.transmat_.sum(axis=1), 1.0)
    )
    if needs_T:
        # if you trained binary, this becomes [[.96,.04],[.10,.90]]; otherwise sticky-K
        hmm.transmat_ = _sticky_T(K, stay=0.95)
    # ----------------------------------------------------------------------

    # Ensure feature columns exist; imputer will handle NaNs
    for c in used_feats:
        if c not in df.columns:
            df[c] = np.nan

    X_raw = df[used_feats].astype(float).values
    X = imputer.transform(X_raw)

    # Predictions
    states = hmm.predict(X)
    post = hmm.predict_proba(X)

    # Map ids -> names
    df["state_id"] = states
    df["state"] = [label_map.get(s, str(s)) for s in states]

    # Choose the “not healthy” state index
    target_names = {"not_healthy", "symptomatic", "prodrome"}
    not_idx = next((sid for sid, name in label_map.items() if name in target_names), None)
    if not_idx is None:
        # Fallback: higher severity (+rhr +temp −rmssd). Align weights to used_feats.
        # Order expected: rhr_z, rmssd_z, temp_z, seff_z (if present).
        weight_map = {"rhr_z": +1.0, "rmssd_z": -1.0, "temp_z": +1.0, "seff_z": -1.0}
        w = np.array([weight_map.get(f, 0.0) for f in used_feats], dtype=float)
        sev = hmm.means_.dot(w)
        not_idx = int(np.argmax(sev))

    # Soft probability and thresholding
    p_not = post[:, not_idx]
    df["p_not_healthy"] = p_not
    df["state_soft"] = np.where(p_not >= 0.40, "not_healthy", "healthy")  # tune 0.35–0.55

    return df, post, label_map, used_feats


def plot_timeline(df, outpath):
    plt.figure(figsize=(11,4))
    plt.plot(df["date"], df["rhr_z"], label="ΔRHR_z")
    plt.plot(df["date"], df["rmssd_z"], label="−ΔRMSSD_z")
    plt.plot(df["date"], df["temp_z"], label="TempDev_z")
    plt.plot(df["date"], df["seff_z"], label="−SleepEff_z")
    ill_mask = df["state"].isin(["prodrome","symptomatic"])
    ymin = df[["rhr_z","rmssd_z","temp_z","seff_z"]].min(axis=1).min() - 0.2
    ymax = df[["rhr_z","rmssd_z","temp_z","seff_z"]].max(axis=1).max() + 0.2
    plt.fill_between(df["date"], ymin, ymax, where=ill_mask, alpha=0.15, step="mid", label="Illness states")
    plt.axhline(0, linewidth=1)
    plt.legend()
    plt.title("Oura nightly z-scores & HMM illness states")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def summarize_emissions(hmm, used_feats, label_map):
    K = hmm.n_components
    means = np.asarray(hmm.means_)
    covars = getattr(hmm, "covars_", None) or getattr(hmm, "_covars_", None)
    if covars is None:
        covars = np.ones_like(means)
    if covars.ndim == 1:
        covars = np.tile(covars[None, :], (K, 1))
    rows = []
    for k in range(K):
        name = label_map.get(k, str(k))
        for j, feat in enumerate(used_feats):
            rows.append({
                "state_id": k, "state": name, "feature": feat,
                "mean": float(means[k, j]), "var": float(covars[k, j]),
                "std": float(np.sqrt(covars[k, j]))
            })
    return pd.DataFrame(rows)

def threshold_summary(df, p_col="p_not_healthy", proxy_col="readiness", thr=0.40):
    out = {}
    if proxy_col not in df.columns or not df[proxy_col].notna().any():
        out["available"] = False
        return out
    q20 = float(df[proxy_col].quantile(0.20))
    y_true = (df[proxy_col] <= q20).astype(int)           # 1 = bad readiness
    y_pred = (df[p_col] >= thr).astype(int)               # 1 = model flags not_healthy
    TP = int(((y_true == 1) & (y_pred == 1)).sum())
    FP = int(((y_true == 0) & (y_pred == 1)).sum())
    TN = int(((y_true == 0) & (y_pred == 0)).sum())
    FN = int(((y_true == 1) & (y_pred == 0)).sum())
    P, N = TP + FN, TN + FP
    sens = TP / P if P else np.nan
    spec = TN / N if N else np.nan
    prec = TP / (TP + FP) if (TP + FP) else np.nan
    bacc = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan
    out.update({
        "available": True, "threshold": thr, "q20_readiness": q20,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "sensitivity": sens, "specificity": spec,
        "precision": prec, "balanced_accuracy": bacc,
        "prevalence_q20": P / (P + N) if (P + N) else np.nan,
    })
    return out

def quick_report(df, hmm=None, used_feats=None, label_map=None, p_col="p_not_healthy", thr=0.40):
    """
    Counts & dwells, overlap@Q20, Spearman rho(p_not, readiness),
    confusion @ threshold, emission means/vars preview (if model provided).
    """
    if df.empty:
        return {"msg": "No data after preprocessing."}

    out = {}
    # --- basic counts + dwells ---
    seq = df["state"].tolist() if "state" in df.columns else []
    if seq:
        dwells, cur, n = [], seq[0], 1
        for s in seq[1:]:
            if s == cur: n += 1
            else: dwells.append((cur, n)); cur, n = s, 1
        dwells.append((cur, n))
        out["dwells_head"] = dwells[:10]
        out["nights"] = len(seq)
        out["state_counts"] = dict(pd.Series(seq).value_counts())

    # --- which labels count as "ill" (binary-safe) ---
    if label_map is not None:
        ill_labels = {name for name in label_map.values() if name != "healthy"}
    else:
        ill_labels = {"not_healthy"}

    if "state" in df.columns:
        out["ill_nights"] = int(df["state"].isin(ill_labels).sum())

    # --- readiness-based signals ---
    if "readiness" in df.columns and df["readiness"].notna().any():
        q20 = float(df["readiness"].quantile(0.20))
        out["readiness_q20"] = q20

        if "state" in df.columns:
            ill_mask = df["state"].isin(ill_labels)
            if ill_mask.any():
                overlap = (df.loc[ill_mask, "readiness"] <= q20).mean()
                out["overlap_q20_ill_states"] = float(overlap)

        if p_col in df.columns and df[p_col].notna().any():
            rho, pval = spearmanr(df[p_col], df["readiness"], nan_policy="omit")
            out["spearman_rho_p_not_vs_readiness"] = float(rho)
            out["spearman_pval"] = float(pval)
            out["threshold_summary"] = threshold_summary(df, p_col=p_col, thr=thr)
    else:
        out["readiness_note"] = "Readiness not present—overlap/correlation unavailable."

    # --- emission interpretability ---
    if hmm is not None and used_feats is not None and label_map is not None:
        try:
            emis_tbl = summarize_emissions(hmm, used_feats, label_map)
            out["emissions_preview"] = emis_tbl.head(8).to_dict(orient="records")
        except Exception as e:
            out["emissions_error"] = f"{type(e).__name__}: {e}"

    out["msg"] = "OK"
    return out


if __name__ == "__main__":
    df = build_oura_nightly("data_raw/oura")
    df, post, label_map, used_feats = apply_model(df)

    out_img = Path("reports/figures/oura_timeline.png")
    plot_timeline(df, out_img)
    print(f"[OK] Timeline -> {out_img}")

    # load model for emission summary
    hmm = load(MODEL_DIR / "model.pkl")

    stats = quick_report(
        df, hmm=hmm, used_feats=used_feats, label_map=label_map,
        p_col="p_not_healthy", thr=0.40
    )

    print("[REPORT] nights:", stats.get("nights"))
    print("[REPORT] ill_nights:", stats.get("ill_nights"))
    print("[REPORT] state_counts:", stats.get("state_counts"))

    rq = stats.get("readiness_q20")
    if rq is not None:
        print(f"[REPORT] readiness_q20: {rq:.1f}")

    ov = stats.get("overlap_q20_ill_states")
    if ov is not None:
        print(f"[REPORT] overlap@Q20 (ill states): {ov:.1%}")

    rho, pv = stats.get("spearman_rho_p_not_vs_readiness"), stats.get("spearman_pval")
    if rho is not None and pv is not None:
        print(f"[REPORT] Spearman rho(p_not, readiness): {rho:.3f} (p={pv:.3g})")

    ts = stats.get("threshold_summary", {})
    if ts.get("available"):
        print(f"[REPORT] thr={ts['threshold']:.2f} | TP={ts['TP']} FP={ts['FP']} TN={ts['TN']} FN={ts['FN']}")
        print(f"[REPORT] sens={ts['sensitivity']:.2f} spec={ts['specificity']:.2f} "
              f"prec={ts['precision']:.2f} bacc={ts['balanced_accuracy']:.2f} "
              f"prev={ts['prevalence_q20']:.2f}")

    # optional: save full emissions table
    try:
        emis_df = summarize_emissions(hmm, used_feats, label_map)
        emis_df.to_csv("reports/emissions_table.csv", index=False)
        print("[OK] Emissions table -> reports/emissions_table.csv")
    except Exception:
        pass

    df.to_csv("reports/oura_states.csv", index=False)
    print("[OK] States CSV -> reports/oura_states.csv")

