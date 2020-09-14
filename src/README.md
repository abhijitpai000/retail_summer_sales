## Documentation

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_dataset() | Performs pre-processing | raw_file_name | train_set, train_clean, test_set, test_clean, selected_data, dropped_data & ord_encoder.pkl | train_clean, test_clean
| train | train_model() | Trains Linear Regression model using 10-folds cv | -- | summer_sales_estimator.pkl | cv results
| predict | test_model() | Predicts on the test_clean dataset | -- | -- | (r2, mae) test scores
                                                                                  
                                                
                                               
       
