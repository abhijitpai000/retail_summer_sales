"""Configuration File"""

TARGET_COLS = 'units_sold'

CAT_COLS = ['product_variation_size_id',
            'product_id']

NUM_COLS = ['rating_three_count',
            'rating_five_count',
            'merchant_rating_count',
            'uses_ad_boosts',
            'discount',
            'discount_percent',
            'badge_fast_shipping',
            'retail_price']


SELECTED_FEATURES = ['rating_three_count',
                     'rating_five_count',
                     'merchant_rating_count',
                     'uses_ad_boosts',
                     'discount',
                     'discount_percent',
                     'product_variation_size_id',
                     'badge_fast_shipping',
                     'product_id',
                     'retail_price',
                     'units_sold']   # <--- TARGET COLUMN.
