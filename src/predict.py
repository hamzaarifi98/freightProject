"""
Load the saved model and create the submission files.

Run after train.py:
    python src/predict.py

Writes validation_predictions.csv (load_id,predicted_rate) and fills the
predicted_rate column of data/december_chart_inputs.csv.
"""
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import build_city_coords, build_daily_index, clean, prepare

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load_artifacts():
    model = joblib.load(MODELS / "lgbm_model.joblib")
    prep = joblib.load(MODELS / "preprocess.joblib")
    return model, prep


def predict(model, X):
    return pd.Series(model.predict(X).clip(min=1), index=X.index).round(2)


def main():
    model, prep = load_artifacts()

    val = clean(pd.read_csv(DATA / "validation.csv", index_col="load_id"))
    chart_raw = pd.read_csv(DATA / "december-chart-inputs.csv")
    chart = clean(chart_raw)

    # Lookups: saved ones from training, plus what validation adds
    # (its own cities and its own dates for the market index)
    coords = pd.concat([prep["coords"], build_city_coords(val)])
    coords = coords[~coords.index.duplicated()]
    daily_index = pd.concat([prep["daily_index"], build_daily_index(val)])
    daily_index = daily_index[~daily_index.index.duplicated()]

    X_val = prepare(val, prep["fill"], daily_index, coords)
    X_chart = prepare(chart, prep["fill"], daily_index, coords)
    for name, X in [("validation", X_val), ("chart", X_chart)]:
        assert list(X.columns) == prep["features"], f"columns differ in {name}"
        assert X.isnull().sum().sum() == 0, f"nulls in {name}"

    # Validation predictions: exactly load_id,predicted_rate
    template = pd.read_csv(DATA / "validation-predictions-template.csv")
    template["predicted_rate"] = template["load_id"].map(predict(model, X_val))
    assert len(template) == 12_000 and template["predicted_rate"].notna().all()
    template.to_csv(ROOT / "validation_predictions.csv", index=False)

    # December chart: keep the original 7 columns, fill only predicted_rate
    chart_raw["predicted_rate"] = predict(model, X_chart).values
    chart_raw.to_csv(DATA / "december_chart_inputs.csv", index=False)

    print("Saved validation_predictions.csv and data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
