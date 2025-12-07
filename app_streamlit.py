import json, ast, io
import datetime as dt
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
import requests  # --- NEW ---
from validate_oura import apply_model, _to_day_datetime
from streamlit_autorefresh import st_autorefresh  


st.set_page_config(page_title="Illness HMM (Oura)", layout="wide")

@st.cache_resource
def load_model():
    hmm = load("models/model.pkl")
    with open("models/labels.json") as f:
        lbl = {int(k): v for k, v in json.load(f).items()}
    return hmm, lbl


# --- NEW: helper to (optionally) update HMM with new data ----------
def update_model_with_df(hmm, df, used_feats):
    """
    Simple re-fit of the HMM on the latest data.
    For GaussianHMM from hmmlearn this will update the parameters in place.
    """
    X = df[used_feats].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if len(X) >= hmm.n_components:
        hmm.fit(X)
    return hmm


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
            if df.shape[1] == 1 and sep == ";":
                continue
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _parse_contributors(s):
    if pd.isna(s):
        return {}
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return {}


def build_oura_from_upload(
    sleep_file=None,
    readiness_file=None,
    temp_file=None,
    hr_file=None,
    resilience_file=None,
):
    # ----- Sleep -----
    if sleep_file is None:
        sleep = pd.DataFrame(columns=["date", "sleep_efficiency", "sleep_score"])
    else:
        sleep = _read_df(sleep_file)
        # Map date-like column
        if "timestamp" in sleep.columns:
            sleep["date"] = _to_day_datetime(sleep["timestamp"])
        else:
            dcol = next(
                (c for c in ["day", "date", "summary_date"] if c in sleep.columns), None
            )
            if dcol is None:
                sleep["date"] = pd.NaT
            else:
                sleep["date"] = _to_day_datetime(sleep[dcol])

        # Sleep efficiency from contributors JSON
        if "contributors" in sleep.columns:
            contrib = sleep["contributors"].apply(_parse_contributors)
            sleep["sleep_efficiency"] = (
                contrib.apply(lambda d: d.get("efficiency", np.nan)) / 100.0
            )
        else:
            sleep["sleep_efficiency"] = np.nan

        if "score" in sleep.columns:
            sleep["sleep_score"] = sleep["score"]
        elif "sleep_score" not in sleep.columns:
            sleep["sleep_score"] = np.nan

        sleep = (
            sleep[["date", "sleep_efficiency", "sleep_score"]]
            .groupby("date", as_index=False)
            .mean()
        )

    # ----- Readiness -----
    if readiness_file is None:
        read = pd.DataFrame(columns=["date", "readiness"])
    else:
        read = _read_df(readiness_file)
        if "timestamp" in read.columns:
            read["date"] = _to_day_datetime(read["timestamp"])
        else:
            dcol = next(
                (c for c in ["day", "date", "summary_date"] if c in read.columns), None
            )
            read["date"] = _to_day_datetime(read[dcol]) if dcol else pd.NaT
        if "readiness" not in read.columns:
            read["readiness"] = read["score"] if "score" in read.columns else np.nan
        read = read[["date", "readiness"]].groupby("date", as_index=False).mean()

    # ----- Temperature deviation -----
    if temp_file is None:
        tdf = pd.DataFrame(columns=["date", "temp_dev"])
    else:
        tdf = _read_df(temp_file)
        if "timestamp" in tdf.columns:
            tdf["date"] = _to_day_datetime(tdf["timestamp"])
        else:
            dcol = next(
                (c for c in ["day", "date", "summary_date"] if c in tdf.columns), None
            )
            tdf["date"] = _to_day_datetime(tdf[dcol]) if dcol else pd.NaT
        if "temperature_deviation" in tdf.columns:
            tdf = tdf.rename(columns={"temperature_deviation": "temp_dev"})
        elif "temperature_delta" in tdf.columns:
            tdf = tdf.rename(columns={"temperature_delta": "temp_dev"})
        elif "temp_deviation" in tdf.columns:
            tdf = tdf.rename(columns={"temp_deviation": "temp_dev"})
        else:
            tdf["temp_dev"] = np.nan
        tdf = tdf[["date", "temp_dev"]].groupby("date", as_index=False).mean()

    # ----- Resting heart rate -----
    if hr_file is None:
        hrt = pd.DataFrame(columns=["date", "rhr"])
    else:
        hrt = _read_df(hr_file)
        if "timestamp" in hrt.columns:
            hrt["date"] = _to_day_datetime(hrt["timestamp"])
        else:
            dcol = next(
                (c for c in ["day", "date", "summary_date"] if c in hrt.columns), None
            )
            hrt["date"] = _to_day_datetime(hrt[dcol]) if dcol else pd.NaT
        if "resting_heart_rate" not in hrt.columns:
            cand = [
                c
                for c in hrt.columns
                if "resting" in c.lower() and "heart" in c.lower()
            ]
            hrt["resting_heart_rate"] = hrt[cand[0]] if cand else np.nan
        hrt = (
            hrt.rename(columns={"resting_heart_rate": "rhr"})[["date", "rhr"]]
            .groupby("date", as_index=False)
            .min()
        )

    # ----- Resilience / HRV (optional) -----
    if resilience_file is None:
        resil = pd.DataFrame(columns=["date", "rmssd"])
    else:
        resil = _read_df(resilience_file)
        if "timestamp" in resil.columns:
            resil["date"] = _to_day_datetime(resil["timestamp"])
        else:
            dcol = next(
                (c for c in ["day", "date", "summary_date"] if c in resil.columns), None
            )
            resil["date"] = _to_day_datetime(resil[dcol]) if dcol else pd.NaT
        if "average_hrv" in resil.columns:
            resil = resil.rename(columns={"average_hrv": "rmssd"})
        else:
            resil["rmssd"] = np.nan
        resil = resil[["date", "rmssd"]].groupby("date", as_index=False).mean()

    # Final merge
    for _df in (sleep, read, tdf, hrt, resil):
        _df["date"] = _to_day_datetime(_df["date"])

    df = (
        sleep.merge(read, on="date", how="left", validate="one_to_one")
        .merge(tdf, on="date", how="left", validate="one_to_one")
        .merge(hrt, on="date", how="left", validate="one_to_one")
        .merge(resil, on="date", how="left", validate="one_to_one")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Build z-scores (same as validator)
    if df["sleep_efficiency"].notna().any() and df["sleep_efficiency"].max() > 1.5:
        df["sleep_efficiency"] = df["sleep_efficiency"] / 100.0

    df["rhr_z"] = trailing_zscore(df["rhr"])
    df["rmssd_z"] = (
        trailing_zscore(df.get("rmssd", pd.Series(index=df.index))) * (-1.0)
    )
    df["temp_z"] = trailing_zscore(df["temp_dev"])
    df["seff_z"] = trailing_zscore(df["sleep_efficiency"]) * (-1.0)

    return df


def plot_timeline(df: pd.DataFrame):
    # Require state column
    if "state" not in df.columns:
        st.warning("No `state` column in DataFrame – cannot draw HMM timeline.")
        return None

    # Require at least one finite value in any z-score column
    zcols = ["rhr_z", "rmssd_z", "temp_z", "seff_z"]
    present_zcols = [c for c in zcols if c in df.columns]
    if not present_zcols:
        st.warning("No z-score columns found – cannot draw HMM timeline.")
        return None

    z_df = df[present_zcols]
    if not np.isfinite(z_df.to_numpy()).any():
        st.warning("Z-score columns contain no finite values – nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(12, 3))

    if "rhr_z" in df.columns:
        ax.plot(df["date"], df["rhr_z"], label="ΔRHR_z")
    if "rmssd_z" in df.columns:
        ax.plot(df["date"], df["rmssd_z"], label="−ΔRMSSD_z")
    if "temp_z" in df.columns:
        ax.plot(df["date"], df["temp_z"], label="TempDev_z")
    if "seff_z" in df.columns:
        ax.plot(df["date"], df["seff_z"], label="−SleepEff_z")

    ill = df["state"].isin(["prodrome", "symptomatic"])

    ymin = z_df.min(axis=1).min() - 0.2
    ymax = z_df.max(axis=1).max() + 0.2

    if np.isfinite([ymin, ymax]).all():
        ax.fill_between(
            df["date"],
            ymin,
            ymax,
            where=ill,
            alpha=0.15,
            label="Illness states",
        )

    ax.axhline(0, linewidth=1)
    ax.legend(loc="upper left")
    ax.set_title("Oura nightly z-scores & HMM states")
    fig.tight_layout()
    return fig



# --- NEW: Live Oura API helpers -----------------------------------
OURA_BASE = "https://api.ouraring.com/v2/usercollection"

def _utc_range(minutes: int):
    """Return (start_iso, end_iso) in UTC for the last `minutes`."""
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(minutes=minutes)

    def fmt(x: dt.datetime) -> str:
        return x.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return fmt(start), fmt(end)


def fetch_live_heartrate(token: str, minutes: int = 15, return_raw: bool = False):
    """Fetch heart rate for the last `minutes` minutes."""
    start_iso, end_iso = _utc_range(minutes)

    url = f"{OURA_BASE}/heartrate"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "start_datetime": start_iso,
        "end_datetime": end_iso,
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)

    try:
        payload = r.json()
    except ValueError:
        payload = {"ERROR": {"message": r.text}}

    df = pd.DataFrame(payload.get("data", []))
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")[["timestamp", "bpm"]]
    else:
        df = pd.DataFrame(columns=["timestamp", "bpm"])

    return (df, payload) if return_raw else df


def fetch_recent_temp_deviation(token: str, days: int = 7, return_raw: bool = False):
    """
    Fetch last `days` of temperature deviation using daily_body_temperature.
    Returns df[date, temp_dev] sorted by date.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    url = f"{OURA_BASE}/daily_readiness"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }

    r = requests.get(url, headers=headers, params=params, timeout=15)
    try:
        payload = r.json()
    except ValueError:
        payload = {"ERROR": {"message": r.text}}

    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        df = pd.DataFrame(columns=["date", "temp_dev"])
    else:
        if "day" in df.columns:
            df["date"] = pd.to_datetime(df["day"])
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "timestamp" in df.columns:
            df["date"] = _to_day_datetime(df["timestamp"])
        else:
            df["date"] = pd.NaT

        if "temperature_deviation" in df.columns:
            df["temp_dev"] = df["temperature_deviation"]
        elif "temperature_delta" in df.columns:
            df["temp_dev"] = df["temperature_delta"]
        elif "temp_deviation" in df.columns:
            df["temp_dev"] = df["temp_deviation"]
        else:
            df["temp_dev"] = np.nan

        df = df[["date", "temp_dev"]].sort_values("date")

    return (df, payload) if return_raw else df


def _oura_get(token: str, endpoint: str, params: dict):
    """Small wrapper that always returns a JSON dict."""
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{OURA_BASE}/{endpoint}", headers=headers, params=params, timeout=15)
    try:
        return r.json()
    except ValueError:
        return {"ERROR": {"message": r.text}}


def _safe_contrib_get(x, key):
    """contributors can be dicts or JSON strings; handle both."""
    if isinstance(x, dict):
        return x.get(key)
    if isinstance(x, str):
        try:
            d = json.loads(x)
            if isinstance(d, dict):
                return d.get(key)
        except Exception:
            return None
    return None


def build_daily_from_api(token: str, days: int = 7, return_raw: bool = False):
    """
    Build a daily feature table for the last `days` days using:
      - daily_sleep        -> sleep_efficiency, sleep_score
      - daily_readiness    -> readiness, resting HR, temperature deviation
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days)

    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    sleep_payload = _oura_get(token, "daily_sleep", params)
    read_payload  = _oura_get(token, "daily_readiness", params)
    # : pull HR timeseries and collapse to daily min/median
    start_iso = f"{start_date.isoformat()}T00:00:00Z"
    end_iso   = f"{(end_date + dt.timedelta(days=1)).isoformat()}T00:00:00Z"
    hr_payload = _oura_get(token, "heartrate", {
        "start_datetime": start_iso,
        "end_datetime":   end_iso,
    })

    hr = pd.DataFrame(hr_payload.get("data", []))

    if not hr.empty and "timestamp" in hr.columns:
        # parse as UTC, then drop tz so it matches sleep/readiness
        hr["timestamp"] = pd.to_datetime(hr["timestamp"], utc=True)

        # normalize to midnight *and* remove timezone
        hr["date"] = (
            hr["timestamp"]
            .dt.tz_convert("UTC")      # keep it in UTC
            .dt.tz_localize(None)      # drop tz info → datetime64[ns]
            .dt.normalize()            # midnight of that day
        )

        hr_daily = (
            hr.groupby("date")["bpm"]
            .median()                   # or median / mean, your choice
            .rename("rhr")
            .reset_index()
        )
    else:
        hr_daily = pd.DataFrame(columns=["date", "rhr"])

    # ---------- daily_sleep ----------
    sleep = pd.DataFrame(sleep_payload.get("data", []))
    if sleep.empty:
        sleep_df = pd.DataFrame(columns=["date", "sleep_efficiency", "sleep_score"])
    else:
        dcol = next((c for c in ("day", "date", "summary_date") if c in sleep.columns), None)
        if dcol is not None:
            sleep["date"] = pd.to_datetime(sleep[dcol])
        else:
            sleep["date"] = pd.NaT

        # sleep efficiency (0–1)
        if "sleep_efficiency" not in sleep.columns:
            if "contributors" in sleep.columns:
                sleep["sleep_efficiency"] = sleep["contributors"].apply(
                    lambda c: _safe_contrib_get(c, "efficiency")
                )
            else:
                sleep["sleep_efficiency"] = np.nan

        # sleep score
        if "sleep_score" not in sleep.columns:
            if "score" in sleep.columns:
                sleep["sleep_score"] = sleep["score"]
            else:
                sleep["sleep_score"] = np.nan

        sleep_df = sleep[["date", "sleep_efficiency", "sleep_score"]]

    # ---------- daily_readiness ----------
    read = pd.DataFrame(read_payload.get("data", []))
    if read.empty:
        read_df = pd.DataFrame(columns=["date", "readiness", "rhr", "temp_dev"])
    else:
        dcol = next((c for c in ("day", "date", "summary_date") if c in read.columns), None)
        if dcol is not None:
            read["date"] = pd.to_datetime(read[dcol])
        else:
            read["date"] = pd.NaT

        # readiness score
        if "readiness" not in read.columns:
            if "score" in read.columns:
                read["readiness"] = read["score"]
            else:
                read["readiness"] = np.nan

        # resting heart rate
        if "resting_heart_rate" in read.columns:
            read["rhr"] = read["resting_heart_rate"]
        else:
            cand = [c for c in read.columns
                    if "resting" in c.lower() and "heart" in c.lower()]
            read["rhr"] = read[cand[0]] if cand else np.nan

        # temperature deviation
        if "temperature_deviation" in read.columns:
            read["temp_dev"] = read["temperature_deviation"]
        elif "temperature_delta" in read.columns:
            read["temp_dev"] = read["temperature_delta"]
        elif "temp_deviation" in read.columns:
            read["temp_dev"] = read["temp_deviation"]
        else:
            read["temp_dev"] = np.nan

        read_df = read[["date", "readiness", "rhr", "temp_dev"]]

    # ---------- merge + sort ----------
    df = (
        sleep_df.merge(read_df, on="date", how="outer", validate="one_to_one")
                .merge(hr_daily, on="date", how="left")
                .sort_values("date")
                .reset_index(drop=True)
    )
    # After merging in hr_daily, resolve duplicate rhr columns
    if "rhr_x" in df.columns and "rhr_y" in df.columns:
        df["rhr"] = df["rhr_y"].fillna(df["rhr_x"])
        df = df.drop(columns=["rhr_x", "rhr_y"])


    # normalize sleep_efficiency if it's 0–100 instead of 0–1
    if "sleep_efficiency" in df.columns:
        if df["sleep_efficiency"].notna().any() and df["sleep_efficiency"].max() > 1.5:
            df["sleep_efficiency"] = df["sleep_efficiency"] / 100.0

    if return_raw:
        return df, {"sleep": sleep_payload, "readiness": read_payload}
    return df



# ------------------------- UI --------------------------------------
st.title("Oura HMM Illness Detector")

tab_batch, tab_live, tab_24h = st.tabs(
    ["Batch CSV (nightly)", "Live HR / Temp", "Last 24h (API → HMM)" ]
)


with tab_batch:
    st.subheader("Upload nightly CSV exports")

    sleep_file = st.file_uploader(
        "Upload Oura sleep.csv (dailysleep.csv)", type=["csv"], key="sleep"
    )
    read_file = st.file_uploader(
        "Upload Oura readiness.csv (dailyreadiness.csv)", type=["csv"], key="readiness"
    )
    temp_file = st.file_uploader(
        "Upload Oura temperature.csv", type=["csv"], key="temp"
    )
    hr_file = st.file_uploader(
        "Upload Oura heartrate.csv", type=["csv"], key="hr"
    )
    res_file = st.file_uploader(
        "Upload Oura dailyresilience.csv (optional)", type=["csv"], key="res"
    )

    if st.button("Run on uploads"):
        df = build_oura_from_upload(sleep_file, read_file, temp_file, hr_file, res_file)
        st.write(f"Rows: {len(df)}")
        st.dataframe(df.head())

        df, post, label_map, used_feats = apply_model(df)
        st.write({"features_used": used_feats})
        st.bar_chart(df["state"].value_counts())

        from validate_oura import plot_timeline as _plot_timeline_file

        out_img = Path("reports/figures/oura_timeline.png")
        _plot_timeline_file(df, out_img)
        st.image(str(out_img))

        # --- NEW: optional HMM update from this dataset -----------
        if st.checkbox("Update HMM with this dataset (session only)"):
            hmm, lbl = load_model()
            hmm = update_model_with_df(hmm, df, used_feats)
            st.success("HMM parameters updated in memory for this session.")

with tab_live:
    st.subheader("Live Heart Rate & Today’s Temperature")

    token = st.text_input(
        "Oura API access token (Bearer)",
        type="password",
        help="Paste the access_token value from the Oura OAuth flow.",
        key="token_live",
    )

    range_label = st.selectbox(
        "Time window",
        ["Last 15 minutes", "Last 1 hour", "Last 4 hours", "Last 24 hours"],
        index=0,
    )
    range_to_minutes = {
        "Last 15 minutes": 15,
        "Last 1 hour": 60,
        "Last 4 hours": 240,
        "Last 24 hours": 1440,
    }
    minutes = range_to_minutes[range_label]

    interval = st.slider(
        "Refresh interval (seconds)", min_value=10, max_value=300, value=30, step=5
    )

    if token:
        st_autorefresh(interval=interval * 1000, key="oura_live_refresh")

        # ---- Heart rate ----
        try:
            hr_df, hr_raw = fetch_live_heartrate(token, minutes, return_raw=True)
            now_utc = dt.datetime.now(dt.timezone.utc)

            if hr_df.empty:
                st.info("No heart-rate samples in the selected time window.")

                hr24_df, _ = fetch_live_heartrate(token, 24 * 60, return_raw=True)
                if hr24_df.empty:
                    st.warning(
                        "API health check: no heart-rate samples in the last 24 hours either.\n\n"
                        "This usually means Oura Cloud has not received any data from your ring yet."
                    )
                else:
                    latest = hr24_df["timestamp"].max()
                    age_minutes = (now_utc - latest).total_seconds() / 60.0
                    st.success(
                        f"API health check: found {len(hr24_df)} samples in the last 24 hours. "
                        f"Most recent sample at: {latest}.\n"
                        f"Oura Cloud appears to be about {age_minutes:.1f} minutes behind real time."
                    )
            else:
                st.success(f"Received {len(hr_df)} heart-rate samples.")
                st.line_chart(hr_df.set_index("timestamp")["bpm"])

            with st.expander("Show raw heart-rate API response"):
                st.json(hr_raw)
        except Exception as e:
            st.error(f"Error fetching heart rate: {e}")

        # ---- Temperature deviation (3 requested behaviors) ----
        temp_df, temp_raw = fetch_recent_temp_deviation(token, days=7, return_raw=True)

        if temp_df.empty:
            st.info(
                "No temperature deviation data in the last 7 days yet.\n\n"
                "Oura only computes temperature deviation once per night after sleep. "
                "You’ll see data here after your next completed sleep and sync."
            )
        else:
            today = dt.date.today()
            today_rows = temp_df[temp_df["date"].dt.date == today]

            if not today_rows.empty:
                row = today_rows.iloc[-1]
                label = "Today’s temperature deviation"
            else:
                row = temp_df.sort_values("date").iloc[-1]
                src_date = row["date"].date()
                label = f"Most recent temperature deviation (from {src_date})"

            if pd.notna(row["temp_dev"]):
                st.metric(label, f"{row['temp_dev']:+.2f} °C")
            else:
                st.info(
                    "Temperature deviation field is present but empty (NaN). "
                    "This can happen if Oura hasn't finished processing last night yet."
                )

            with st.expander("Temperature deviation – last 7 days"):
                st.line_chart(temp_df.set_index("date")["temp_dev"])

        with st.expander("Show raw temperature API response"):
            st.json(temp_raw)
    else:
        st.info("Paste your Oura access token above to start the live view.")

with tab_24h:
    st.subheader("Run HMM on last ~24 hours (daily API data)")

    token_24 = st.text_input(
        "Oura API access token (Bearer)",
        type="password",
        key="token_24h",
    )

    st.caption(
        "This pulls daily sleep / readiness / temperature / HRV from Oura for the last 2 days "
        "and feeds it into the existing illness HMM."
    )

    if token_24:
    # pull a wider window, e.g. last 7 days
        df_api, raw = build_daily_from_api(token_24, days=7, return_raw=True)

        if df_api.empty:
            st.warning(
                "No daily Oura data found for the requested range.\n\n"
                "Check that your ring has synced and Oura has processed your recent sleep."
            )
            with st.expander("Show raw Oura API response"):
                st.json(raw)
        else:
            # Optionally show which date is actually the most recent
            last_date = df_api["date"].max().date()
            st.info(f"Last processed Oura daily data is for {last_date}.")

            st.write("Daily features:")
            st.dataframe(df_api.tail())

            df_hmm, post, label_map, used_feats = apply_model(df_api)
            st.write({"features_used": used_feats})
            st.bar_chart(df_hmm["state"].value_counts())

            # use the local, safe plot_timeline defined above
            fig = plot_timeline(df_hmm)
            if fig is not None:
                st.pyplot(fig)


