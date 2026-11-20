"""Raw transactions -> model-ready frame: one row per product per calendar day.

Steps (previously notebook cells 1-5, 37-39, 42, 44, 46):
load -> daily aggregation -> shift sales to publication days -> continuous calendar ->
calendar features -> lag features -> drop 29 February.
"""
import warnings

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Product", "Net Price", "Quantity"]


def load_raw(path, cfg):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    start, end = pd.Timestamp(cfg["study"]["start"]), pd.Timestamp(cfg["study"]["end"])
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def aggregate_daily(df):
    return (df.groupby(["Product", "Date"], as_index=False)
              .agg({"Net Price": "sum", "Quantity": "sum"})
              .sort_values(["Product", "Date"]).reset_index(drop=True))


def shift_to_publication_days(daily_df, publication_days):
    """Sales recorded on a non-publication day are pushed to the next publication day."""
    unknown = set(daily_df["Product"]) - set(publication_days)
    if unknown:
        raise ValueError(f"No publication schedule for: {sorted(unknown)}")
    out = daily_df.copy()
    for product, valid in publication_days.items():
        m = (out["Product"] == product).to_numpy()
        days_ahead = np.array([min((v - d) % 7 for v in valid) for d in range(7)])   # lookup by weekday
        dow = out.loc[m, "Date"].dt.dayofweek.to_numpy()
        out.loc[m, "Date"] = out.loc[m, "Date"] + pd.to_timedelta(days_ahead[dow], unit="D")
    return (out.groupby(["Product", "Date"], as_index=False)
               .agg({"Net Price": "sum", "Quantity": "sum"})
               .sort_values(["Product", "Date"]).reset_index(drop=True))


def build_calendar(shifted_df, publication_days, study_end, extend_to=None):
    """Continuous daily calendar per product. Non-publication days = 0; publication days with no
    recorded sales = 0 up to the study end. Rows after the study end (production horizon) keep NaN targets.
    Sales shifted beyond the study end are dropped."""
    study_end = pd.Timestamp(study_end)
    frames = []
    for product in sorted(shifted_df["Product"].unique()):
        d = shifted_df[(shifted_df["Product"] == product) & (shifted_df["Date"] <= study_end)]
        if study_end - d["Date"].max() > pd.Timedelta(days=10):
            warnings.warn(f"{product}: last sale on {d['Date'].max().date()}, well before the study end {study_end.date()}")
        end = pd.Timestamp(extend_to) if extend_to is not None else study_end     # every product reaches the study end
        cal = pd.DataFrame({"Date": pd.date_range(d["Date"].min(), end, freq="D")})
        cal["Product"] = product
        cal["Day_of_Week"] = cal["Date"].dt.dayofweek
        cal["Is_Publication_Day"] = cal["Day_of_Week"].isin(publication_days[product]).astype(int)
        cal = cal.merge(d[["Date", "Net Price", "Quantity"]], on="Date", how="left")
        pub = (cal["Is_Publication_Day"] == 1).to_numpy()
        hist = (cal["Date"] <= study_end).to_numpy()
        cal.loc[~pub, ["Net Price", "Quantity"]] = 0.0
        cal.loc[hist & pub, ["Net Price", "Quantity"]] = cal.loc[hist & pub, ["Net Price", "Quantity"]].fillna(0.0)
        frames.append(cal)
    return pd.concat(frames, ignore_index=True).sort_values(["Product", "Date"]).reset_index(drop=True)


def add_calendar_features(df):
    df = df.copy()
    df["Day_of_Week"] = df["Date"].dt.dayofweek
    df["Day_of_Month"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Year"] = df["Date"].dt.year
    df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)
    df["Is_Month_Start"] = df["Date"].dt.is_month_start.astype(int)
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
    df["COVID_Intervention"] = df["Date"].between("2020-03-01", "2021-12-31").astype(int)
    return df


def add_lag_features(df):
    """364-day lags (weekday-aligned) and same-calendar-date-last-year values.
    Computed on the full calendar BEFORE 29 February is removed."""
    df = df.sort_values(["Product", "Date"]).reset_index(drop=True)
    g = df.groupby("Product")
    df["Revenue_Lag_364"] = g["Net Price"].shift(364)
    df["Quantity_Lag_364"] = g["Quantity"].shift(364)
    df["Date_Last_Year"] = df["Date"] - pd.DateOffset(years=1)
    key = pd.MultiIndex.from_arrays([df["Product"], df["Date_Last_Year"]])
    lookup = df.set_index(["Product", "Date"])
    df["Revenue_Same_Date_Last_Year"] = lookup["Net Price"].reindex(key).to_numpy()
    df["Quantity_Same_Date_Last_Year"] = lookup["Quantity"].reindex(key).to_numpy()
    return df


def drop_leap_days(df):
    keep = ~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))
    return df[keep].sort_values(["Product", "Date"]).reset_index(drop=True)


def build_model_frame(raw, cfg, extend_to=None):
    """extend_to: also create future calendar rows (targets NaN) up to this date, for production forecasts."""
    publication_days = cfg["publication_days"]
    study_end = pd.Timestamp(cfg["study"]["end"])
    daily = aggregate_daily(raw)
    shifted = shift_to_publication_days(daily, publication_days)
    cal = build_calendar(shifted, publication_days, study_end, extend_to)
    df = drop_leap_days(add_lag_features(add_calendar_features(cal)))

    # sanity check (previously a notebook cell): revenue and quantity totals are preserved
    hist = df[df["Date"] <= study_end]
    before = daily.groupby("Product")[["Net Price", "Quantity"]].sum()
    after = hist.groupby("Product")[["Net Price", "Quantity"]].sum()
    rel = ((after - before).abs() / before.abs().clip(lower=1e-9)).max().max()
    if rel > 5e-3:
        warnings.warn(f"Revenue/quantity totals changed by {rel:.2%} after shifting (sales shifted past the study end are dropped).")
    return df