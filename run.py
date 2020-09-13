"""Execute src code in command line.

NOTE: After run, use tests.py for quick sanity check.
"""

# Local Package Imports.
from src.preprocess import make_dataset
from src.train import train_model
from src.predict import test_model

import numpy as np

if __name__ == '__main__':
    # Pre-process dataset.
    train_clean, test_clean = make_dataset(raw_file_name="raw.csv")

    # Train.
    cv_results = train_model()

    print("TRAIN RESULTS:"
          f"\nR2: {np.mean(cv_results['test_r2'])}"
          f"\nMAE:{np.mean(cv_results['test_neg_mean_absolute_error'])}")

    # Test.
    test_mae, test_r2 = test_model()
    print("\nTEST RESULTS:"
          f"\nR2: {test_r2}"
          f"\nMAE:{test_mae}")