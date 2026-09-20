"""
Preprocessing for the freight rate model.

Usage from another file:
    from preprocessing import load_and_prepare
    data = load_and_prepare()
    X_train, y_train = data["X_train"], data["y_train"]
"""
import pandas as pd
import numpy as np

TARGET = "posted_rate"
BANDS = [0, 300, 800, 1500, 4000]
FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
    "day_of_week", "month",
    "equipment_Dry Van", "equipment_Flatbed", "equipment_Reefer",
]


# =================================================================
# CLEANING + DATE FEATURES (row-level, safe before the split)
# Dates stay as plain numbers: cyclical encoding was tested and
# increased test error (less than one year of data)
# =================================================================
def clean(d):
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    d.loc[d["weight"] <= 0, "weight"] = np.nan          # weight must be positive
    d["day_of_week"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    return d


# =================================================================
# LOOKUPS FROM FEATURES ONLY (no target involved)
# =================================================================
def build_city_coords(*frames):
    """City name -> (lat, lon), used for the December file (no coordinates)."""
    both = pd.concat(frames)
    return pd.concat([
        both[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        both[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
    ]).drop_duplicates("city").set_index("city")


def build_daily_index(*frames):
    """Date -> median market index, for filling missing values."""
    return pd.concat(frames).groupby("date")["market_index"].median()


# =================================================================
# OUTLIERS (training data only)
# Finds the two empty gaps in the rate-vs-similar-loads ratio and
# drops loads outside them
# =================================================================
def remove_outliers(d, verbose=True):
    d = d.copy()
    d["rpm"] = d[TARGET] / d["distance"]
    d["dist_band"] = pd.cut(d["distance"], BANDS)

    group = d.groupby(["equipment", "dist_band"], observed=True)["rpm"]
    typical = group.median()
    ratio = d["rpm"] / group.transform("median")

    s = ratio.sort_values().reset_index(drop=True)
    jumps = s.diff()
    i = jumps[s <= 1].idxmax()
    j = jumps[s.shift(1) >= 1].idxmax()
    low = (s[i - 1] + s[i]) / 2
    high = (s[j - 1] + s[j]) / 2

    bounds = pd.DataFrame({"min_rpm": typical * low, "max_rpm": typical * high})
    d = d.join(bounds, on=["equipment", "dist_band"])
    keep = d["rpm"].between(d["min_rpm"], d["max_rpm"])

    if verbose:
        print(f"gaps: {s[i-1]:.3f}->{s[i]:.3f} and {s[j-1]:.3f}->{s[j]:.3f} "
              f"| cutoffs {low:.3f}, {high:.3f} | dropped {(~keep).sum()} of {len(d)}")
    return d[keep].drop(columns=["rpm", "dist_band", "min_rpm", "max_rpm"])


# =================================================================
# FILL VALUES: learned from training data only
# =================================================================
def fit_fill_values(d):
    return {
        "weight_medians": d.groupby("equipment")["weight"].median(),
        "quote_median": d["quote_signal"].median(),
    }


# =================================================================
# APPLY FILLS + ENCODE: works on any file, never drops rows
# =================================================================
def prepare(d, fill, daily_index, coords):
    d = d.copy()

    for side in ("pickup", "delivery"):                 # December file has no coordinates
        if f"{side}_lat" not in d:
            d[f"{side}_lat"] = d[side].map(coords["lat"])
            d[f"{side}_lon"] = d[side].map(coords["lon"])

    if "market_index" not in d:                          # December file has no market index
        d["market_index"] = np.nan
    d["market_index"] = d["market_index"].fillna(d["date"].map(daily_index))

    if "quote_signal" not in d:                          # December file has no quote signal
        d["quote_signal"] = np.nan
    d["quote_signal"] = d["quote_signal"].fillna(fill["quote_median"])

    d["weight"] = d["weight"].fillna(d["equipment"].map(fill["weight_medians"]))

    d = pd.get_dummies(d, columns=["equipment"], dtype=int)
    return d.reindex(columns=FEATURES, fill_value=0)


# =================================================================
# FULL PIPELINE
# =================================================================
def load_and_prepare(folder="data", test_share=0.2, verbose=True):
    # Load (load_id as index: kept for the template, never a feature)
    train_test = pd.read_csv(f"{folder}/train-test.csv", index_col="load_id")
    val = pd.read_csv(f"{folder}/validation.csv", index_col="load_id")
    chart = pd.read_csv(f"{folder}/december-chart-inputs.csv")

    train_test, val, chart = clean(train_test), clean(val), clean(chart)

    coords = build_city_coords(train_test, val)
    daily_index = build_daily_index(train_test, val)

    # Time-based split: the most recent dates are the test set
    train_test = train_test.sort_values("date")
    cutoff = train_test["date"].quantile(1 - test_share)
    train = train_test[train_test["date"] < cutoff].copy()
    test = train_test[train_test["date"] >= cutoff].copy()

    # Stage 1: learn from train, evaluate on test (test keeps its outliers)
    train = remove_outliers(train, verbose)
    fill = fit_fill_values(train)
    X_train, y_train = prepare(train, fill, daily_index, coords), train[TARGET]
    X_test, y_test = prepare(test, fill, daily_index, coords), test[TARGET]

    # Stage 2: learn from all labeled data for the final model
    full = remove_outliers(train_test, verbose)
    final_fill = fit_fill_values(full)
    X_full, y_full = prepare(full, final_fill, daily_index, coords), full[TARGET]
    X_val = prepare(val, final_fill, daily_index, coords)          # index = load_id
    X_chart = prepare(chart, final_fill, daily_index, coords)

    # Checks
    for name, X in [("train", X_train), ("test", X_test), ("full", X_full),
                    ("validation", X_val), ("chart", X_chart)]:
        assert X.isnull().sum().sum() == 0, f"nulls in {name}"
        assert list(X.columns) == FEATURES, f"columns differ in {name}"
    assert len(X_val) == len(val), "validation rows were dropped"

    if verbose:
        print(f"split cutoff: {cutoff.date()}")
        for name, X in [("train", X_train), ("test", X_test), ("full", X_full),
                        ("validation", X_val), ("chart", X_chart)]:
            print(f"{name:11s} {X.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_full": X_full, "y_full": y_full,
        "X_val": X_val, "X_chart": X_chart,
        "train": train, "test": test,       # cleaned frames, for baseline and evaluation
    }


if __name__ == "__main__":
    load_and_prepare()