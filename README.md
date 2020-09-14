# E-commerce Summer Sales: How well a product is likely to sell.

### Overview:

Developed an Interpretable ML model to understand how well a product published on the E-Commerce platform Wish is going to sell. In this study, I extracted 10 significant features out of 43 provided in the dataset and trained a Linear Regression model that predicts "units_sold" of products. 

**Evaluation Metrics used:** Mean absolute error and R-Squared


### Findings:

It is observed that out of the 10 features, products marked rating 3 and 5 have a significant positive impact on sales, followed by merchants who used ad boosts, their total rating count, discount percent on the product, with a retail price of the product having a significant negative impact.

**Sales Estimator Equation:**

**units_sold** = *(uses_ad_boosts x 642.80) + (rating_three_count x 15.06) + (rating_five_count x 3.33) +
                 (merchant_rating_count x 0.003) + (product_id x -0.850) + (discount x -1.73) +
                 (discount_percent x -6.038) + (product_variation_size_id x -10.65) +
                 (retail_price x -16.418) + (badge_fast_shipping x -3173.258)*


*Note: Weights for each feature are absolute weights, may differ from relative weights obtained by transforming features to a common scale, which is depicted in the plot below*

**Relative Weights :**

<img src="https://github.com/abhijitpai000/retail_summer_sales/blob/master/report/fw.png" />



### Data Source:

I used [Summer Sales](https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish) data for this analysis.

## Final Report & Package Walk-Through

To reproduce this study, use modules in 'src' directory of this repo. (setup instructions below) and walk-through of the package is presented in the [final report](https://github.com/abhijitpai000/retail_summer_sales/tree/master/report)

## Setup instructions

#### Creating Python environment

This repository has been tested on Python 3.7.6.

1. Cloning the repository:

`git clone https://github.com/abhijitpai000/retail_summer_sales.git`

2. Navigate to the git clone repository.

`cd customer_segmentation_rfm`

3. Install [virtualenv](https://pypi.org/project/virtualenv/)

`pip install virtualenv`

`virtualenv rfm`

4. Activate it by running:

`rfm/Scripts/activate`

5. Install project requirements by using:

`pip install -r requirements.txt`

**Note**
For make_dataset() to work, please place the raw data (.csv from data source) in the 'datasets' directory.
