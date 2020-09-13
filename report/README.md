# E-commerce Summer Clothes Sales

**Overview :**

Developed an Interpretable ML model to understand how well a product published on the E-Commerce platform Wish is going to sell. In this study, I extracted 10 significant features out of 43 provided in the dataset and trained a Linear Regression model that predicts "units_sold" of products. 

**Findings:**


Based on the model, It is observed that out of the 10 features, products marked rating 3 and 5 have a significant positive impact on sales, followed by merchants who used ad boosts, their total rating count, discount percent on the product, with a retail price of the product having a significant negative impact.

**Sales Estimator Equation:**

**Evaluation Metrics:** Mean absolute error and R-Squared score.

**units_sold** = *(uses_ad_boosts x 642.80) + (rating_three_count x 15.06) + (rating_five_count x 3.33) +
                 (merchant_rating_count x 0.003) + (product_id x -0.850) + (discount x -1.73) +
                 (discount_percent x -6.038) + (product_variation_size_id x -10.65) +
                 (retail_price x -16.418) + (badge_fast_shipping x -3173.258)*


*Note: Weights for each feature are absolute weights, may differ from relative weights obtained by transforming features to a common scale, which is depicted in the plot below*

**Relative Weights :**




## Analysis Walk-through

**Table of Contents**

1. [Introduction](#introduction)
2. [Preprocess](#preprocess)
3. [Training](#train)
4. [Prediction](#predict)


```python
# Setting Git Clone Path as Current Working Directory (cwd).

import os
FILE_PATH = "Git/Clone/Path"
os.chdir(FILE_PATH)   # Changes cwd
os.getcwd()   # Prints cwd
```

# Introduction <a name="introduction"></a>

**Codebase Structure** 'src' directory.

| Module | Function | Description |
| :--- | :--- | :--- |
| preprocess | make_dataset() | Performs pre-processing returns train_clean and test_clean datasets.
| train | train_model() | Trains Linear Regression model, returns cross validation scores.
| predict | test_model() | Predicts on the test_clean dataset and returns test scores.


```python
# Local Imports.
from src.preprocess import make_dataset
from src.train import train_model
from src.predict import test_model

# Analysis.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Joblib
import joblib
```


```python
# Loading Raw data.

raw = pd.read_csv("datasets/raw.csv")

raw.shape
```




    (1573, 43)




```python
# Checking top 3 rows.

with pd.option_context("display.max_rows", 4, "display.max_columns", 50):
    display(raw.head(2))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>title_orig</th>
      <th>price</th>
      <th>retail_price</th>
      <th>currency_buyer</th>
      <th>units_sold</th>
      <th>uses_ad_boosts</th>
      <th>rating</th>
      <th>rating_count</th>
      <th>rating_five_count</th>
      <th>rating_four_count</th>
      <th>rating_three_count</th>
      <th>rating_two_count</th>
      <th>rating_one_count</th>
      <th>badges_count</th>
      <th>badge_local_product</th>
      <th>badge_product_quality</th>
      <th>badge_fast_shipping</th>
      <th>tags</th>
      <th>product_color</th>
      <th>product_variation_size_id</th>
      <th>product_variation_inventory</th>
      <th>shipping_option_name</th>
      <th>shipping_option_price</th>
      <th>shipping_is_express</th>
      <th>countries_shipped_to</th>
      <th>inventory_total</th>
      <th>has_urgency_banner</th>
      <th>urgency_text</th>
      <th>origin_country</th>
      <th>merchant_title</th>
      <th>merchant_name</th>
      <th>merchant_info_subtitle</th>
      <th>merchant_rating_count</th>
      <th>merchant_rating</th>
      <th>merchant_id</th>
      <th>merchant_has_profile_picture</th>
      <th>merchant_profile_picture</th>
      <th>product_url</th>
      <th>product_picture</th>
      <th>product_id</th>
      <th>theme</th>
      <th>crawl_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020 Summer Vintage Flamingo Print  Pajamas Se...</td>
      <td>2020 Summer Vintage Flamingo Print  Pajamas Se...</td>
      <td>16.0</td>
      <td>14</td>
      <td>EUR</td>
      <td>100</td>
      <td>0</td>
      <td>3.76</td>
      <td>54</td>
      <td>26.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Summer,Fashion,womenunderwearsuit,printedpajam...</td>
      <td>white</td>
      <td>M</td>
      <td>50</td>
      <td>Livraison standard</td>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>50</td>
      <td>1.0</td>
      <td>Quantité limitée !</td>
      <td>CN</td>
      <td>zgrdejia</td>
      <td>zgrdejia</td>
      <td>(568 notes)</td>
      <td>568</td>
      <td>4.128521</td>
      <td>595097d6a26f6e070cb878d1</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/5e9ae51d43d6a96e303acdb0</td>
      <td>https://contestimg.wish.com/api/webimage/5e9ae...</td>
      <td>5e9ae51d43d6a96e303acdb0</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SSHOUSE Summer Casual Sleeveless Soirée Party ...</td>
      <td>Women's Casual Summer Sleeveless Sexy Mini Dress</td>
      <td>8.0</td>
      <td>22</td>
      <td>EUR</td>
      <td>20000</td>
      <td>1</td>
      <td>3.45</td>
      <td>6135</td>
      <td>2269.0</td>
      <td>1027.0</td>
      <td>1118.0</td>
      <td>644.0</td>
      <td>1077.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Mini,womens dresses,Summer,Patchwork,fashion d...</td>
      <td>green</td>
      <td>XS</td>
      <td>50</td>
      <td>Livraison standard</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>50</td>
      <td>1.0</td>
      <td>Quantité limitée !</td>
      <td>CN</td>
      <td>SaraHouse</td>
      <td>sarahouse</td>
      <td>83 % avis positifs (17,752 notes)</td>
      <td>17752</td>
      <td>3.899673</td>
      <td>56458aa03a698c35c9050988</td>
      <td>0</td>
      <td>NaN</td>
      <td>https://www.wish.com/c/58940d436a0d3d5da4e95a38</td>
      <td>https://contestimg.wish.com/api/webimage/58940...</td>
      <td>58940d436a0d3d5da4e95a38</td>
      <td>summer</td>
      <td>2020-08</td>
    </tr>
  </tbody>
</table>
</div>


# Data Pre-processing <a name="preprocess"></a>

**Experiment:**
* *Goal:* To build an interpretable model, I chose the family of Linear Models, which produces a mathematical equation and is computationally efficient..


* *Feature Engineering:* To gain more insights from the dataset I developed the following features.
    * Discount Price = (retail_price) - (price)
    * Discount % = (discount)/(price) x 100
    * Markup % = (retail_price - price)/(price) x 100
    * Average Merchant Rating
    * Average Product Rating 
    * Number of Tags marked by each merchant on the product.
    
    
* *Feature Selection:* Using techniques such as L1 regularization, Drop Feature Selection, Residual Plot Analysis, and Correlations, I derived the following 10 significant features out of 48 (includes engineered) which are impacting the "units_sold" target.
    * 'rating_three_count'
    * 'rating_five_count'
    * 'merchant_rating_count'
    * 'uses_ad_boosts'
    * 'discount'
    * 'discount_percent'
    * 'product_variation_size_id'
    * 'badge_fast_shipping'
    * 'product_id'
    * 'retail_price'
    
**make_dataset()** Based on the insights gained from experiment, make dataset will perform following actions.
1. Feature Engineering
2. Extract selected features, drops the rest.
3. Split data into train_set and test_set 
4. Process both the sets and produce train_clean, test_clean.
5. Saves the datasets produced at each step in the "datasets" directory and saves encoder used in the "models" directory.


```python
# Extracting Train Clean and Test Clean sets.

train_clean, test_clean = make_dataset(raw_file_name="raw.csv")

train_clean.shape, test_clean.shape   # Includes target column.
```




    ((1179, 11), (394, 11))




```python
train_clean.head(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_variation_size_id</th>
      <th>product_id</th>
      <th>rating_three_count</th>
      <th>rating_five_count</th>
      <th>merchant_rating_count</th>
      <th>uses_ad_boosts</th>
      <th>discount</th>
      <th>discount_percent</th>
      <th>badge_fast_shipping</th>
      <th>retail_price</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>93.0</td>
      <td>1273.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>127</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>6</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56.0</td>
      <td>1101.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>10600</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>5</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>335.0</td>
      <td>124.0</td>
      <td>590.0</td>
      <td>5534</td>
      <td>0</td>
      <td>-3.35</td>
      <td>0.0</td>
      <td>0</td>
      <td>9</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56.0</td>
      <td>733.0</td>
      <td>98.0</td>
      <td>227.0</td>
      <td>5985</td>
      <td>1</td>
      <td>-75.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>89</td>
      <td>5000</td>
    </tr>
  </tbody>
</table>
</div>



# Train Model <a name="train"></a>

**train_model()**
* Fits a fine-tuned Linear Regression model to train_clean dataset.
* Performs 10 folds Cross-validation.
* Returns CV results


```python
# CV SCores of Trained Linear Regression Model.

cv_results = train_model()
```


```python
cv_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_neg_mean_absolute_error</th>
      <th>test_r2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.009973</td>
      <td>0.000996</td>
      <td>-2442.741900</td>
      <td>0.716343</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.001995</td>
      <td>0.001996</td>
      <td>-1994.564402</td>
      <td>0.711172</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005020</td>
      <td>0.001004</td>
      <td>-1881.891736</td>
      <td>0.839834</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.002986</td>
      <td>0.000995</td>
      <td>-1425.309827</td>
      <td>0.798542</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.001960</td>
      <td>0.000998</td>
      <td>-1524.038103</td>
      <td>0.833808</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000997</td>
      <td>0.001999</td>
      <td>-2077.143283</td>
      <td>0.758954</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.001992</td>
      <td>0.001995</td>
      <td>-2326.109729</td>
      <td>0.807365</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.001995</td>
      <td>0.000998</td>
      <td>-1733.161344</td>
      <td>0.790995</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.001995</td>
      <td>0.000997</td>
      <td>-1351.601507</td>
      <td>0.759605</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001995</td>
      <td>0.000999</td>
      <td>-1704.415536</td>
      <td>0.659981</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Mean R2 and Mean MAE (absolute_error)

np.mean(cv_results['test_r2']), np.mean(-cv_results['test_neg_mean_absolute_error'])
```




    (0.7676600419460257, 1846.0977368351098)



# Test Prediction <a name="test"></a>

**test_model()**
* Predicts on the test_clean dataset.
* Returns R2 and MAE of predictions.


```python
test_r2, test_mae = test_model()
```


```python
# TEST SCORES. 

test_mae, test_r2   # TRAINING SCORES: (0.7676600419460257, 1846.0977368351098)
```




    (0.8535742464800788, 2015.8739238541998)



End
