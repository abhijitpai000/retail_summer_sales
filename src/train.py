"""Train Linear Regression Model using train_model()"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
import pandas as pd
from pathlib import Path
import joblib

from .config import TARGET_COLS


def train_model():
    """
    Trains Linear Regression Model.

    Yields
    ------
        cv_results: 'models/cv_results.csv'
        Trained model: 'models/summer_sales_estimator.pkl'
    """
    train_clean = pd.read_csv("datasets/train_clean.csv")

    features = train_clean.drop(TARGET_COLS, axis=1)
    target = train_clean[TARGET_COLS]

    lin_reg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    cv_results = cross_validate(lin_reg, features, target,
                                scoring=['neg_mean_absolute_error', 'r2'],
                                cv=kfold)
    # Save CV Results.
    cv_path = Path.cwd() / "models/cv_results.csv"
    cv_results = pd.DataFrame(cv_results)
    cv_results.to_csv(cv_path, index=False)

    # Save Summer Sales Estimator.
    lin_reg.fit(features, target)
    est_path = Path.cwd() / "models/summer_sales_estimator.pkl"
    joblib.dump(lin_reg, est_path)
    return cv_results
