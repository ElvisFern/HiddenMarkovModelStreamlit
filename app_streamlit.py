# app_streamlit.py
import json,ast,io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from validate_oura import apply_model, _to_day_datetime

st.set_page_config(page_title="Illness HMM (Oura)", layout="wide")

@st.cache_resource
def load_model():
    hmm = load("models/model.pkl")
    with open("models/labels.json") as f:
        lbl = {int(k): v for k, v in json.load(f).items()}
    return hmm, lbl

def trailing_zscore(s, win=30, minp=7):
    m = s.rolling(win, min_periods=minp).mean().shift(1)
    sd = s.rolling(win, min_periods=minp).std(ddof=0).shift(1)
    return (s - m) / sd

def _read_df(file) -> pd.DataFrame:
    """Robust CSV reader for Oura exports (semicolon first, then comma)."""
    b = file.getvalue() if hasattr(file, "getvalue") else file.read()
    for sep in (";", ","):
        try:
            df = pd.read_csv(io.BytesIO(b), sep=sep, encoding="utf-8-sig")
            # If it collapsed into a single column, try next sep
            if df.shape[1] == 1 and sep == ";":
                continue
            return df
        except Exception:
            continue
    return pd.DataFrame()

def _parse_contributors(s):
    if pd.isna(s): return {}
    try: return json.loads(s)
    except Exception:
        try: return ast.literal_eval(s)
        except Exception: return {}

def build_oura_from_upload(sleep_file=None, readiness_file=None, temp_file=None, hr_file=None, resilience_file=None):
    # ----- Sleep -----
    if sleep_file is None:
        sleep = pd.DataFrame(columns=["date","sleep_efficiency","sleep_score"])
    else:
        sleep = _read_df(sleep_file)
        # Map date-like column
        if "timestamp" in sleep.columns:
            sleep["date"] = _to_day_datetime(sleep["timestamp"])
        else:
            dcol = next((c for c in ["day","date","summary_date"] if c in sleep.columns), None)
            if dcol is None: sleep["date"] = pd.NaT
            else: sleep["date"] = _to_day_datetime(sleep[dcol])

        # Sleep efficiency from contributors JSON
        if "contributors" in sleep.columns:
            contrib = sleep["contributors"].apply(_parse_contributors)
            sleep["sleep_efficiency"] = contrib.apply(lambda d: d.get("efficiency", np.nan)) / 100.0
        else:
            sleep["sleep_efficiency"] = np.nan

        if "score" in sleep.columns:
            sleep["sleep_score"] = sleep["score"]
        elif "sleep_score" not in sleep.columns:
            sleep["sleep_score"] = np.nan

        sleep = (sleep[["date","sleep_efficiency","sleep_score"]]
                 .groupby("date", as_index=False).mean())

    # ----- Readiness -----
    if readiness_file is None:
        read = pd.DataFrame(columns=["date","readiness"])
    else:
        read = _read_df(readiness_file)
        if "timestamp" in read.columns:
            read["date"] = _to_day_datetime(read["timestamp"])
        else:
            dcol = next((c for c in ["day","date","summary_date"] if c in read.columns), None)
            read["date"] = _to_day_datetime(read[dcol]) if dcol else pd.NaT
        if "readiness" not in read.columns:
            read["readiness"] = read["score"] if "score" in read.columns else np.nan
        read = read[["date","readiness"]].groupby("date", as_index=False).mean()

    # ----- Temperature deviation -----
    if temp_file is None:
        tdf = pd.DataFrame(columns=["date","temp_dev"])
    else:
        tdf = _read_df(temp_file)
        if "timestamp" in tdf.columns:
            tdf["date"] = _to_day_datetime(tdf["timestamp"])
        else:
            dcol = next((c for c in ["day","date","summary_date"] if c in tdf.columns), None)
            tdf["date"] = _to_day_datetime(tdf[dcol]) if dcol else pd.NaT
        if "temperature_deviation" in tdf.columns:  tdf = tdf.rename(columns={"temperature_deviation":"temp_dev"})
        elif "temperature_delta" in tdf.columns:    tdf = tdf.rename(columns={"temperature_delta":"temp_dev"})
        elif "temp_deviation" in tdf.columns:       tdf = tdf.rename(columns={"temp_deviation":"temp_dev"})
        else:                                        tdf["temp_dev"] = np.nan
        tdf = tdf[["date","temp_dev"]].groupby("date", as_index=False).mean()

    # ----- Resting heart rate -----
    if hr_file is None:
        hrt = pd.DataFrame(columns=["date","rhr"])
    else:
        hrt = _read_df(hr_file)
        if "timestamp" in hrt.columns:
            hrt["date"] = _to_day_datetime(hrt["timestamp"])
        else:
            dcol = next((c for c in ["day","date","summary_date"] if c in hrt.columns), None)
            hrt["date"] = _to_day_datetime(hrt[dcol]) if dcol else pd.NaT
        if "resting_heart_rate" not in hrt.columns:
            # try a fallback column, else NaN
            cand = [c for c in hrt.columns if "resting" in c.lower() and "heart" in c.lower()]
            hrt["resting_heart_rate"] = hrt[cand[0]] if cand else np.nan
        hrt = hrt.rename(columns={"resting_heart_rate":"rhr"})[["date","rhr"]] \
                 .groupby("date", as_index=False).min()

    # ----- Resilience / HRV (optional) -----
    if resilience_file is None:
        resil = pd.DataFrame(columns=["date","rmssd"])
    else:
        resil = _read_df(resilience_file)
        if "timestamp" in resil.columns:
            resil["date"] = _to_day_datetime(resil["timestamp"])
        else:
            dcol = next((c for c in ["day","date","summary_date"] if c in resil.columns), None)
            resil["date"] = _to_day_datetime(resil[dcol]) if dcol else pd.NaT
        if "average_hrv" in resil.columns:
            resil = resil.rename(columns={"average_hrv":"rmssd"})
        else:
            resil["rmssd"] = np.nan
        resil = resil[["date","rmssd"]].groupby("date", as_index=False).mean()

    # Final merge (all dates are datetime64[ns] at midnight)
    for _df in (sleep, read, tdf, hrt, resil):
        _df["date"] = _to_day_datetime(_df["date"])

    df = (sleep.merge(read, on="date", how="left", validate="one_to_one")
               .merge(tdf,  on="date", how="left", validate="one_to_one")
               .merge(hrt,  on="date", how="left", validate="one_to_one")
               .merge(resil,on="date", how="left", validate="one_to_one")
               .sort_values("date").reset_index(drop=True))

    # Build z-scores (same as validator)
    def trailing_zscore(s, win=30, minp=7):
        m = s.rolling(win, min_periods=minp).mean().shift(1)
        sd = s.rolling(win, min_periods=minp).std(ddof=0).shift(1)
        return (s - m) / sd

    if df["sleep_efficiency"].notna().any() and df["sleep_efficiency"].max() > 1.5:
        df["sleep_efficiency"] = df["sleep_efficiency"] / 100.0

    df["rhr_z"]   = trailing_zscore(df["rhr"])
    df["rmssd_z"] = trailing_zscore(df.get("rmssd", pd.Series(index=df.index))) * (-1.0)
    df["temp_z"]  = trailing_zscore(df["temp_dev"])
    df["seff_z"]  = trailing_zscore(df["sleep_efficiency"]) * (-1.0)

    return df

def plot_timeline(df):
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(df["date"], df["rhr_z"], label="ΔRHR_z")
    ax.plot(df["date"], df["rmssd_z"], label="−ΔRMSSD_z")
    ax.plot(df["date"], df["temp_z"], label="TempDev_z")
    ax.plot(df["date"], df["seff_z"], label="−SleepEff_z")
    ill = df["state"].isin(["prodrome","symptomatic"])
    ymin = df[["rhr_z","rmssd_z","temp_z","seff_z"]].min(axis=1).min() - 0.2
    ymax = df[["rhr_z","rmssd_z","temp_z","seff_z"]].max(axis=1).max() + 0.2
    ax.fill_between(df["date"], ymin, ymax, where=ill, alpha=0.15, step="mid", label="Illness states")
    ax.axhline(0, linewidth=1)
    ax.legend(loc="upper left")
    ax.set_title("Oura nightly z-scores & HMM states")
    fig.tight_layout()
    return fig

st.title("Oura HMM Illness Detector")

sleep_file = st.file_uploader("Upload Oura sleep.csv (dailysleep.csv)", type=["csv"])
read_file  = st.file_uploader("Upload Oura readiness.csv (dailyreadiness.csv)", type=["csv"])
temp_file  = st.file_uploader("Upload Oura temperature.csv", type=["csv"])
hr_file    = st.file_uploader("Upload Oura heartrate.csv", type=["csv"])
res_file   = st.file_uploader("Upload Oura dailyresilience.csv (optional)", type=["csv"])

if st.button("Run"):
    df = build_oura_from_upload(sleep_file, read_file, temp_file, hr_file, res_file)
    st.write(f"Rows: {len(df)}")
    st.dataframe(df.head())

    df, post, label_map, used_feats = apply_model(df)
    st.write({"features_used": used_feats})
    st.bar_chart(df["state"].value_counts())

    # Optional: save plot via your existing plot_timeline and show it
    from validate_oura import plot_timeline
    out_img = Path("reports/figures/oura_timeline.png")
    plot_timeline(df, out_img)
    st.image(str(out_img))
