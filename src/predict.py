"""Predict on test_set using Trained Model using test_predict"""

import joblib
import pandas as pd

# Local Imports.
from .config import TARGET_COLS
from sklearn.metrics import mean_absolute_error, r2_score


def test_model():
    """
    Tests Trained model in test_clean.csv

    Returns
    -------
        test_r2, test_mae (tuple)
    """
    test_set = pd.read_csv('datasets/test_clean.csv')

    test_y = test_set[TARGET_COLS]
    test_X = test_set.drop(TARGET_COLS, axis=1)

    summer_sales_estimator = joblib.load("models/summer_sales_estimator.pkl")
    y_preds = summer_sales_estimator.predict(test_X)

    mae = mean_absolute_error(test_y, y_preds)
    r2 = r2_score(test_y, y_preds)
    return mae, r2
