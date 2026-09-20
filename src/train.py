"""
Train the final LightGBM model and save it with its preprocessing values.

Run from anywhere:
    python src/train.py

Hyperparameters come from the tuning in src/train.ipynb
(randomized search with time-series cross-validation).
"""
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from preprocessing import FEATURES, load_and_prepare, remove_outliers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

PARAMS = dict(
    n_estimators=234,
    learning_rate=0.0264891626801987,
    num_leaves=92,
    min_child_samples=17,
    subsample=0.7035119926400067,
    subsample_freq=1,
    colsample_bytree=0.7760609974958406,
    objective="l2",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def evaluate(model, X_test, y_test, test):
    """Test error on all loads and on loads without corrupted rates."""
    pred = pd.Series(model.predict(X_test), index=X_test.index)
    clean = remove_outliers(test, verbose=False).index
    print(f"Test MAE (all):    ${mean_absolute_error(y_test, pred):.1f}")
    print(f"Test MAE (clean):  ${mean_absolute_error(y_test[clean], pred[clean]):.1f}")
    print(f"Test RMSE (clean): ${root_mean_squared_error(y_test[clean], pred[clean]):.1f}")


def main():
    data = load_and_prepare(folder=str(DATA))

    # Stage 1: train on Jan-Aug, evaluate on Sep-Oct
    model = LGBMRegressor(**PARAMS)
    model.fit(data["X_train"], data["y_train"])
    evaluate(model, data["X_test"], data["y_test"], data["test"])

    # Stage 2: retrain on all labeled data with the same settings
    final_model = LGBMRegressor(**PARAMS)
    final_model.fit(data["X_full"], data["y_full"])

    # Save the model and everything needed to prepare new data the same way
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS / "lgbm_model.joblib")
    joblib.dump(
        {
            "fill": data["final_fill"],          # weight medians, quote signal median
            "coords": data["coords"],            # city -> lat/lon
            "daily_index": data["daily_index"],  # date -> median market index
            "features": FEATURES,                # column order the model expects
        },
        MODELS / "preprocess.joblib",
    )
    print(f"Saved model and preprocessing values to {MODELS}")


if __name__ == "__main__":
    main()
