# Freight Rate Prediction

Predicting the posted rate (in dollars) of truckload freight loads, built for the Spotter ML Engineer assessment.

The model is trained on 48,000 labeled loads from January to October 2025. It predicts:

- the rate of 12,000 unlabeled loads from November and December 2025 (`validation_predictions.csv`), and
- the rate of one fixed lane (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) for every day of December 2025.

Training and prediction are two separate scripts: `src/train.py` fits the model and saves it, and `src/predict.py` loads the saved model and writes the submission files.

## Quick start

```bash
python -m pip install -r requirements.txt

python src/train.py     # fit the model, save it to models/
python src/predict.py   # load it, write the predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Both scripts can be run from any folder. The trained model is already committed in `models/`, so `python src/predict.py` works on its own without retraining.

## Results

The final model is a LightGBM regressor. It was evaluated on a held-out test set made of the most recent two months of labeled data (2025-08-31 to 2025-10-31), split by date rather than at random.

| Model | CV MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Baseline (typical rate per mile × distance) | n/a | $92.0 | $126.3 |
| **LightGBM (selected)** | **$73.6** | $56.4 | **$83.7** |
| Random Forest | $81.7 | $56.0 | $84.4 |
| Gradient boosting (scikit-learn) | $86.4 | $56.3 | $85.0 |
| XGBoost | $78.6 | $60.4 | $90.7 |

- **CV MAE** is time-series cross-validation inside the training period. It was the selection criterion.
- **Test MAE / RMSE** are computed on test loads without corrupted rates (see [Data quality](#data-quality)). On *all* test loads, corrupted ones included, MAE is $112 to $116 for the four models.
- LightGBM was picked on cross-validation error, not on test error. The test scores of the top three models are within a dollar of each other.
- `src/train.py` reproduces the LightGBM row: Test MAE $56.4 and RMSE $83.7 on clean loads, $111.8 MAE on all loads.

For the fixed December lane the model predicts **$835.64 to $843.87** (about $2.33 per mile), following a weekly pattern driven by the market index.

## Repository layout

```
.
├── data/
│   ├── train-test.csv                      # 48,000 labeled loads (Jan–Oct 2025)
│   ├── validation.csv                      # 12,000 loads to predict (Nov–Dec 2025)
│   ├── validation-predictions-template.csv # load_id list to fill in
│   ├── december-chart-inputs.csv           # fixed December lane, empty predicted_rate
│   └── december_chart_inputs.csv           # same file with predicted_rate filled (output)
├── src/
│   ├── preprocessing.py                    # cleaning, outlier removal, feature building
│   ├── train.py                            # fit the final model and save it
│   ├── predict.py                          # load the saved model and write predictions
│   └── train.ipynb                         # model tuning and comparison (research)
├── models/
│   ├── lgbm_model.joblib                   # trained LightGBM model
│   └── preprocess.joblib                   # fill values and lookups needed to prepare new data
├── notebook/                               # exploratory analysis (optional)
│   ├── trainEDA.ipynb
│   ├── validationEDA.ipynb
│   ├── decemberEDA.ipynb
│   └── validation-pred-template.ipynb
├── score.py                                # provided validator and December chart script
├── scorer_results/candidate_december.png   # December chart produced by score.py
├── validation_predictions.csv              # final predictions (output)
├── report.pdf                              # written report
├── freight-rate-ml-assessment.pdf          # original assessment instructions
└── requirements.txt
```

## Setup

Python 3.10 or newer (developed on Python 3.13).

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter   # only needed to run the notebooks
```

`requirements.txt` does not pin scikit-learn, LightGBM or XGBoost, so a fresh install picks up recent versions. `train.py` uses `sklearn.metrics.root_mean_squared_error`, so it needs scikit-learn 1.4 or newer.

The input CSVs are already in `data/`. If you replace them, keep the file names shown in the layout above.

## Pipeline

```
data/train-test.csv ─┐
data/validation.csv ─┼─► preprocessing.py ─► train.py ─► models/lgbm_model.joblib
                     │                                   models/preprocess.joblib
                     │                                            │
data/validation.csv ─┼─► preprocessing.py ─► predict.py ◄─────────┘
data/december-…csv  ─┘                           │
                                                 ├─► validation_predictions.csv
                                                 └─► data/december_chart_inputs.csv
```

### 1. Train: `python src/train.py`

1. Loads and preprocesses the data with `load_and_prepare` from `src/preprocessing.py`.
2. **Stage 1, evaluation.** Fits LightGBM on the training period (before 2025-08-31) and prints test MAE and RMSE on the held-out two months.
3. **Stage 2, final fit.** Refits the same model with the same hyperparameters on **all** labeled data.
4. Saves two files to `models/`:
   - `lgbm_model.joblib`, the fitted model, and
   - `preprocess.joblib`, everything needed to prepare new data the same way: weight medians per equipment type, the quote signal median, the city → coordinates lookup, the date → market index lookup, and the feature column order.

Expected output:

```
Test MAE (all):    $111.8
Test MAE (clean):  $56.4
Test RMSE (clean): $83.7
Saved model and preprocessing values to .../models
```

The hyperparameters are set in `PARAMS` at the top of `train.py`. They came from the randomized search in `src/train.ipynb`; `train.py` does not tune anything, so it runs in seconds. Random seeds are fixed (`random_state=42`).

### 2. Predict: `python src/predict.py`

1. Loads the model and preprocessing values from `models/`.
2. Cleans `validation.csv` and `december-chart-inputs.csv`.
3. Extends the saved lookups with what the validation file adds: its own cities (for coordinates) and its own dates (for the market index).
4. Prepares both files with the saved fill values and checks that the columns match the model's features and that there are no nulls.
5. Writes:
   - `validation_predictions.csv` in the project root, with exactly `load_id,predicted_rate` for all 12,000 loads, and
   - `data/december_chart_inputs.csv`, the December template with `predicted_rate` filled in.

Predictions are clipped to at least $1 and rounded to cents.

### 3. Validate: `score.py`

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` checks the file formats, row counts, ids and the fixed December inputs, then saves the chart to `scorer_results/candidate_december.png`. It does not compute accuracy; final validation metrics are calculated by Spotter after submission.

### Preprocessing on its own

```bash
cd src
python preprocessing.py
```

Prints the outlier gaps, split date and feature matrix shapes.

### Tuning and model comparison

`src/train.ipynb` is where the four models were tuned, compared and the LightGBM settings chosen. It is not needed to train or predict. To re-run it from the terminal:

```bash
cd src
jupyter nbconvert --to notebook --execute --inplace train.ipynb
```

It takes a few minutes and imports `preprocessing.py` from its own folder, so run it with `src/` as the working directory. If you re-tune and get new hyperparameters, copy them into `PARAMS` in `train.py`.

The EDA notebooks in `notebook/` are optional and independent of the pipeline. They document what I found in the data, and some read their CSVs with relative paths, so adjust the paths if you re-run them from a different folder.

## Approach

### Split and validation

The validation set comes after all labeled data in time, so the labeled data is split by date to match: the last 20% of dates (from 2025-08-31) is the test set and everything before is training data. A random split would put loads from the same days in both sets and overstate accuracy.

Four models (LightGBM, XGBoost, Random Forest and scikit-learn `HistGradientBoostingRegressor`) were tuned with randomized search and 3-fold `TimeSeriesSplit` cross-validation on the training period, scored by MAE. The test set was used once to evaluate the tuned models. The chosen model is then retrained with the same hyperparameters on **all** labeled data before predicting the validation loads.

All fill values and outlier cutoffs are learned from training data only, and validation rows are never dropped.

### Data quality

- **Invalid weights.** Weights of zero or less are treated as missing and filled with the median weight of the same equipment type.
- **Missing market index.** Filled with the median index of the other loads on the same date, since it is a daily market measure.
- **Corrupted rates (about 1.4%).** Relative to similar loads (same equipment and distance band), normal rates sit between about 0.81× and 1.34× the typical rate per mile. Corrupted ones fall below about 0.47× or above about 2.14×, with nothing in between, and no feature explains them. The cutoffs are placed in the middle of those two empty gaps and the loads are removed from **training data only**. The test set keeps them, and metrics are reported both ways.
- **Unseen cities.** Eight cities in the validation set never appear in the labeled data (about 12% of validation loads), so locations are represented by coordinates rather than by city name.
- **December file.** It has no coordinates, market index or quote signal. Coordinates are looked up by city, the market index is the median of validation loads on that date, and the quote signal is set to the training median.

### Features

13 features: pickup and delivery latitude/longitude, distance, weight, market index, quote signal, day of week, month, and one-hot equipment type (Dry Van, Flatbed, Reefer). Dates are kept as plain numbers because cyclical encoding increased test error with less than a year of data. `load_id` is only kept as an index and is never a feature.

## Limitations

- The labeled data ends in October, so the model has no knowledge of holidays or seasonal effects in November and December. It treats them like October.
- Feature and model comparisons were made on the same test set, so the reported test error may be slightly optimistic.
- The December chart uses a market index taken from the validation loads and a median quote signal, since the December file has neither.
- `predict.py` expects the same input columns as the training files. A city that is in neither the labeled data nor the validation file has no coordinates, so the null check in `predict.py` stops the run until its coordinates are added.
