import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def load_train_cv_results():
    cv_results = pd.read_csv("models/cv_results.csv")
    r2 = np.mean(cv_results["test_r2"])
    mae = np.mean(cv_results["test_neg_mean_absolute_error"])
    return r2, -mae

def test_r2_score(load_train_cv_results):
    assert load_train_cv_results[0] > 0.70, "Attained r2 score < 0.70"

def test_mae_score(load_train_cv_results):
    assert load_train_cv_results[1] < 1900, "Attained mae score > 1900"
