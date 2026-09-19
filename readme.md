#Freight Rate Prediction

Predicting the posted rate for truckload freight loads, built for the Spotter ML Engineer assessment.

The model is trained on 48,000 labeled loads from January to October 2025 and predicts 12,000 loads from November and December 2025, plus a fixed lane for every day of December.

#Results

The final model is LightGBM. On a held-out test set of the most recent two months (Aug 31 to Oct 31), it has a mean absolute error of $56.4 per load on normal loads, compared with $92.0 for a rate-per-mile baseline.


#Setup

Python 3.10 or newer.

bash
python -m pip install -r requirements.txt

Place the four provided files in data/ with these exact names:

data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv



How to run

1. Train the model and create the predictions. Open notebook/train.ipynb and run all cells, or run it from the terminal:

bash
jupyter nbconvert --to notebook --execute --inplace notebook/train.ipynb

This loads and preprocesses the data, tunes the four models (a few minutes, depending on the machine), evaluates them, retrains LightGBM on all labeled data, and writes:

validation_predictions.csv in the project root
the predicted_rate column of data/december_chart_inputs.csv


2. Validate the outputs and create the December chart:

bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

The chart is saved to scorer_results/candidate_december.png.

The EDA notebooks are optional and independent of training; they document what I found in the data.