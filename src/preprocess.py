"""Pre-Processing Module: make_dataset()

Split Data
----------
- Splits data into train and test set.

Drop Data :
-----
- Drop Filter Columns
    - create a dropped copy
    - create a filtered copy

Feature Engineering.
-------------------
selc_cols = ['rating_three_count',
             'rating_five_count',
             'merchant_rating_count',
             'uses_ad_boosts',
             'discount',
             'discount_percent',
             'product_variation_size_id',
             'badge_fast_shipping',
             'product_id',
             'retail_price',
             'units_sold']
Cleaning
--------
- rating_cols = ['rating_five_count', 'rating_three_count']
    - fill missing with zero.
- Discount = retail_price - price
    - Negative values are markup.
- Discount Percent = Discount/retail x 100
    - fill discount_percent < 0 = 0
- product_variation_size_id
    - fill missing with NONE.
"""

# Libraries.
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Path and Serializing models.
from pathlib import Path
import joblib

# Local Imports
from .config import TARGET_COLS, CAT_COLS, NUM_COLS, SELECTED_FEATURES


# STEP 1
def _drop_data(raw_file_name):
    """
    Extracts significant features for modelling and drops rest.

    Parameters
    ----------
        raw_file_name: str, with .csv extension.

    Yields
    ------
       dropped_data: 'datasets/dropped_data.csv'

    Returns
    -------
        selected_data + feature engineered.
    """
    raw_df = pd.read_csv(f"datasets/{raw_file_name}")

    # Select data.
    fe_data = _feature_engineer(raw_df)
    selected_data = fe_data[SELECTED_FEATURES]
    select_path = Path.cwd() / "datasets/selected_data.csv"
    selected_data.to_csv(select_path, index=False)

    # Drop data.
    drop_path = Path.cwd() / "datasets/dropped_data.csv"
    dropped_data = fe_data.drop(SELECTED_FEATURES, axis=1)
    dropped_data.to_csv(drop_path, index=False)

    return selected_data


# STEP 2
def _feature_engineer(dataframe):
    """
    Steps
    -----
    - rating_cols = ['rating_five_count', 'rating_three_count']
        - fill missing with zero.
    - Discount = retail_price - price
        - Negative values are markup.
    - Discount Percent = Discount/retail x 100
        - fill discount_percent < 0 = 0
    - product_variation_size_id
        - fill missing with NONE.

    Returns
    -------
        fe_data.
    """
    # Rating Columns.
    rating_cols = ['rating_five_count', 'rating_four_count',
                   'rating_three_count', 'rating_two_count',
                   'rating_one_count']

    dataframe[rating_cols] = dataframe[rating_cols].fillna(0, axis=0)

    # Discount and Discount Percent.
    dataframe['discount'] = dataframe['price'] - dataframe['retail_price']
    dataframe['discount_percent'] = (dataframe['discount'] / dataframe['price']) * 100
    dataframe['discount_percent'] = dataframe['discount_percent'].clip(lower=0)

    # Product Variation size id.
    dataframe['product_variation_size_id'] = dataframe['product_variation_size_id'].fillna("NONE", axis=0)
    return dataframe


# STEP 3
def _split_data(dataframe, shuffle=True):
    """
    Split data into train and test set (75:25 ratio).

    Parameters
    ----------
        dataframe: pandas dataframe.

        shuffle: bool, default=True.
            Shuffles data while splitting.
    Yields
    ------
        train_set: 'datasets/train_set'
        test_set: 'datasets/test_set'
    """
    train, test = train_test_split(dataframe,
                                   shuffle=shuffle,
                                   random_state=0)

    # Saving File.
    train_path = Path.cwd() / "datasets/train_set.csv"
    test_path = Path.cwd() / "datasets/test_set.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return


# STEP 4
def _train_clean():
    """
    Ordinal Encoding for Train Categorical Columns.

    NOTE:
    - Fits on entire dataset categorical features (to avoid 'unseen data' error).
    - Transforms train_set categorical only.

    Yields
    ------
        train_clean: 'datasets/trian_clean.csv'
        Ordinal Encoder: 'models/ord_encoder.pkl'

    Returns
    -------
        train_clean.
    """
    # Fit Categorical with Ordinal Encoder.
    full_data = pd.read_csv("datasets/selected_data.csv")

    full_cat_features = full_data[CAT_COLS]

    ord_encoder = OrdinalEncoder()
    ord_encoder.fit(full_cat_features)
    pkl_path = Path.cwd() / "models/ord_encoder.pkl"
    joblib.dump(ord_encoder, pkl_path)  # Saving ordinal encoder.

    # Transform Train set.
    train_set = pd.read_csv('datasets/train_set.csv')

    cat_data = train_set[CAT_COLS]
    num_data = train_set[NUM_COLS]
    target = train_set[TARGET_COLS]

    # Ordinal Encoding.
    cat_encoded_data = pd.DataFrame(ord_encoder.transform(cat_data),
                                    index=cat_data.index,
                                    columns=cat_data.columns)

    train_clean = pd.concat([cat_encoded_data, num_data, target], axis=1)
    clean_path = Path.cwd() / "datasets/train_clean.csv"
    train_clean.to_csv(clean_path, index=False)
    return train_clean

def _test_clean():
    """
    Ordinal Encoding for Test Categorical Columns.

    Yields
    ------
        test_clean: 'datasets/test_clean.csv'

    Returns
    -------
        test_clean.
    """
    test_set = pd.read_csv('datasets/test_set.csv')

    cat_data = test_set[CAT_COLS]
    num_data = test_set[NUM_COLS]
    target = test_set[TARGET_COLS]

    ord_encoder = joblib.load("models/ord_encoder.pkl")

    # Ordinal Encoding.
    cat_encoded_data = pd.DataFrame(ord_encoder.transform(cat_data),
                                    index=cat_data.index,
                                    columns=cat_data.columns)

    test_clean = pd.concat([cat_encoded_data, num_data, target], axis=1)
    clean_path = Path.cwd() / "datasets/test_clean.csv"
    test_clean.to_csv(clean_path, index=False)
    return test_clean

def make_dataset(raw_file_name):
    """
    Pre-Processes data.

    Parameters
    ----------
        raw_file_name: str, with .csv.

    Yields
    ------
        dropped_data: 'datasets/dropped_data.csv'
        selected_data: 'datasets/selected_data.csv'
        train_set: 'datasets/train_set.csv'
        train_clean: 'datasets/train_clean.csv'
        test_set: 'datasets/test_set.csv'

    Returns
    -------
        train_clean, test_clean.
    """

    # Extracting features.
    selected_data = _drop_data(raw_file_name=raw_file_name)

    # Creating Splits.
    _split_data(selected_data, shuffle=True)

    # Pre-Processing Train set and test set.
    train_clean = _train_clean()
    test_clean = _test_clean()

    return train_clean, test_clean

