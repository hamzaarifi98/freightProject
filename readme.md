# Freight Rate Prediction

Predicting the posted rate (in dollars) of truckload freight loads, built for the Spotter ML Engineer assessment.

The model is trained on 48,000 labeled loads from January to October 2025. It predicts:

- the rate of 12,000 unlabeled loads from November and December 2025 (`validation_predictions.csv`), and
- the rate of one fixed lane (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) for every day of December 2025.

The full write-up is in [report.pdf](report.pdf).

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
│   └── train.ipynb                         # tuning, evaluation, final fit, predictions
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

## Approach

### Split and validation

The validation set comes after all labeled data in time, so the labeled data is split by date to match: the last 20% of dates (from 2025-08-31) is the test set and everything before is training data. A random split would put loads from the same days in both sets and overstate accuracy.

Four models (LightGBM, XGBoost, Random Forest and scikit-learn `HistGradientBoostingRegressor`) were tuned with randomized search and 3-fold `TimeSeriesSplit` cross-validation on the training period, scored by MAE. The test set was used once to evaluate the tuned models. The chosen model was then retrained with the same hyperparameters on **all** labeled data before predicting the validation loads.

All fill values and outlier cutoffs are learned from training data only, and validation rows are never dropped.

### Data quality

- **Invalid weights.** Weights of zero or less are treated as missing and filled with the median weight of the same equipment type.
- **Missing market index.** Filled with the median index of the other loads on the same date, since it is a daily market measure.
- **Corrupted rates (about 1.4%).** Relative to similar loads (same equipment and distance band), normal rates sit between about 0.81× and 1.34× the typical rate per mile. Corrupted ones fall below about 0.47× or above about 2.14×, with nothing in between, and no feature explains them. The cutoffs are placed in the middle of those two empty gaps and the loads are removed from **training data only**. The test set keeps them, and metrics are reported both ways.
- **Unseen cities.** Eight cities in the validation set never appear in the labeled data (about 12% of validation loads), so locations are represented by coordinates rather than by city name.
- **December file.** It has no coordinates, market index or quote signal. Coordinates are looked up by city, the market index is the median of validation loads on that date, and the quote signal is set to the training median.

### Features

13 features: pickup and delivery latitude/longitude, distance, weight, market index, quote signal, day of week, month, and one-hot equipment type (Dry Van, Flatbed, Reefer). Dates are kept as plain numbers because cyclical encoding increased test error with less than a year of data. `load_id` is only kept as an index and is never a feature.

## Setup

Python 3.10 or newer (developed on Python 3.13).

```bash
python -m pip install -r requirements.txt
python -m pip install jupyter   # only needed to run the notebooks from the terminal
```

`train.ipynb` uses `sklearn.metrics.root_mean_squared_error`, so it needs scikit-learn 1.4 or newer. `requirements.txt` does not pin it, so a fresh install picks up a recent version.

The input CSVs are already in `data/`. If you replace them, keep the file names shown in the layout above.

## How to run

**1. Train and predict.** Open `src/train.ipynb` and run all cells, or run it from the terminal:

```bash
cd src
jupyter nbconvert --to notebook --execute --inplace train.ipynb
```

The notebook imports `preprocessing.py` from its own folder and reads data from `../data`, so run it with `src/` as the working directory. It:

1. loads and preprocesses the data (`load_and_prepare` in `src/preprocessing.py`),
2. computes the rate-per-mile baseline,
3. tunes the four models (a few minutes in total, depending on the machine),
4. evaluates them on the test set,
5. retrains LightGBM on all labeled data, and
6. writes two outputs:
   - `validation_predictions.csv` in the project root, and
   - `data/december_chart_inputs.csv`, the December template with `predicted_rate` filled in.

Predictions are clipped to at least $1 and rounded to cents. Random seeds are fixed (`random_state=42`).

You can also run the preprocessing on its own, which prints the outlier gaps, split date and feature matrix shapes:

```bash
cd src
python preprocessing.py
```

**2. Validate the outputs and make the December chart.**

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`score.py` checks the file formats, row counts, ids and the fixed December inputs, then saves the chart to `scorer_results/candidate_december.png`. It does not compute accuracy; final validation metrics are calculated by Spotter after submission.

The EDA notebooks in `notebook/` are optional and independent of training. They document what I found in the data, and some read their CSVs with relative paths, so adjust the paths if you re-run them from a different folder.

## Limitations

- The labeled data ends in October, so the model has no knowledge of holidays or seasonal effects in November and December. It treats them like October.
- Feature and model comparisons were made on the same test set, so the reported test error may be slightly optimistic.
- The December chart uses a market index taken from the validation loads and a median quote signal, since the December file has neither.
