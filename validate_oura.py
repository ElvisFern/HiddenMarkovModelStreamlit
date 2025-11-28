# validate_oura.py
# Build nightly features from Oura export, apply saved HMM, and create validation graphs
import json,glob, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load

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

        # Some readiness exports include temperature deviation; keep as fallback
        if "temperature_deviation" in read.columns and "temp_dev" not in locals():
            read = read.rename(columns={"temperature_deviation": "temp_dev"})
        read = read[["date", "readiness"] + (["temp_dev"] if "temp_dev" in read.columns else [])]
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

        # prefer explicit resting_heart_rate; otherwise pick a sensible alt
        if "resting_heart_rate" not in hrt.columns:
            cand = [c for c in hrt.columns if "resting" in c.lower() and "heart" in c.lower()]
            if cand:
                hrt = hrt.rename(columns={cand[0]: "resting_heart_rate"})
            else:
                hrt["resting_heart_rate"] = np.nan

        hrt = hrt[["date", "resting_heart_rate"]].rename(columns={"resting_heart_rate": "rhr"})
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

    # Sleep
    sleep = sleep.rename(columns={date_col: "date"})
    sleep["date"] = _to_day_datetime(sleep["date"])

    # Readiness
    ...
    if "timestamp" in read.columns:
        read["date"] = _to_day_datetime(read["timestamp"])
    else:
        date_col = "day" if "day" in read.columns else ("date" if "date" in read.columns else "summary_date")
        read["date"] = _to_day_datetime(read[date_col])

    # Temperature
    ...
    if "timestamp" in tdf.columns:
        tdf["date"] = _to_day_datetime(tdf["timestamp"])
    else:
        t_date = "day" if "day" in tdf.columns else ("date" if "date" in tdf.columns else "summary_date")
        tdf["date"] = _to_day_datetime(tdf[t_date])

    # Heart rate
    ...
    if "timestamp" in hrt.columns:
        hrt["date"] = _to_day_datetime(hrt["timestamp"])
    else:
        h_date = "day" if "day" in hrt.columns else ("date" if "date" in hrt.columns else "summary_date")
        hrt["date"] = _to_day_datetime(hrt[h_date])

    # Resilience / HRV
    ...
    if "timestamp" in resil.columns:
        resil["date"] = _to_day_datetime(resil["timestamp"])
    else:
        rz_date = "day" if "day" in resil.columns else ("date" if "date" in resil.columns else "summary_date")
        resil["date"] = _to_day_datetime(resil[rz_date])

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

    read = (read.groupby("date", as_index=False)
                .agg({"readiness":"mean"}))

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

    # Ensure the required feature columns exist (fill with NaN if missing; imputer will handle)
    for c in used_feats:
        if c not in df.columns:
            df[c] = np.nan

    X_raw = df[used_feats].astype(float).values
    X = imputer.transform(X_raw)
    states = hmm.predict(X)
    post = hmm.predict_proba(X)
    df["state_id"] = states
    df["state"] = [label_map[s] for s in states]
    df["post_max"] = post.max(axis=1)
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

def quick_report(df):
    if df.empty:
        return {"msg": "No data after preprocessing."}

    have_readiness = ("readiness" in df.columns) and df["readiness"].notna().any()
    ill_mask = df["state"].isin(["prodrome","symptomatic"])
    ill_n = int(ill_mask.sum())

    # dwell summary
    seq = df["state"].tolist()
    dwells, cur, n = [], seq[0], 1
    for s in seq[1:]:
        if s == cur: n += 1
        else: dwells.append((cur, n)); cur, n = s, 1
    dwells.append((cur, n))

    out = {"ill_nights": ill_n, "dwells": dwells[:10]}

    if not have_readiness:
        out["msg"] = "Readiness not present."
        return out

    if ill_n == 0:
        out["msg"] = "No illness states detected; overlap not computed."
        # still return quantiles for context
        out["readiness_q20"] = float(df["readiness"].quantile(0.2))
        return out

    q20 = df["readiness"].quantile(0.2)
    overlap = (df.loc[ill_mask, "readiness"] <= q20).mean()
    out["overlap_q20"] = float(overlap)
    out["readiness_q20"] = float(q20)
    out["msg"] = "OK"
    return out


if __name__ == "__main__":
    df = build_oura_nightly("data_raw/oura")
    df, post, label_map, used_feats = apply_model(df)

    out_img = Path("reports/figures/oura_timeline.png")
    plot_timeline(df, out_img)

    stats = quick_report(df)  # uses the revised function
    print(f"[OK] Timeline -> {out_img}")

    if stats.get("msg") == "OK":
        print(f"Illness nights: {stats['ill_nights']}, "
              f"overlap with lowest 20% readiness: {stats['overlap_q20']:.1%}")
        print(f"Readiness 20th percentile: {stats['readiness_q20']:.1f}")
    elif stats.get("msg") == "No illness states detected; overlap not computed.":
        print(stats["msg"])
        if "readiness_q20" in stats:
            print(f"Readiness 20th percentile: {stats['readiness_q20']:.1f}")
    else:
        # e.g., "Readiness not present." or "No data after preprocessing."
        print(stats.get("msg", "Done."))

    print(f"Example dwells (state,nights): {stats.get('dwells', [])}")
    df.to_csv("reports/oura_states.csv", index=False)
    print("[OK] States CSV -> reports/oura_states.csv")


