# Customer Purchase Prediction with PyTorch

An end to end machine learning pipeline that predicts whether an e-commerce customer will make a purchase based on browsing behavior. Built for a case study with RetailTech Solutions, an international e-commerce platform, to support personalization and revenue growth.

## Overview

The project has three stages:

1. **Clean** raw session data (missing values, data types, allowed categories).
2. **Prepare** model features (MinMax scaling and one hot encoding).
3. **Model** purchases with a PyTorch neural network and validate it on held out data.

## Data

Each row is one customer session.

| Column | Type | Rule |
|---|---|---|
| `customer_id` | Integer | Unique, no missing values |
| `time_spent` | Float | Minutes on site, missing values filled with the median |
| `pages_viewed` | Integer | Pages viewed, missing values filled with the mean (rounded) |
| `basket_value` | Float | Value of items in basket, missing values filled with 0 |
| `device_type` | String | Mobile, Desktop or Tablet, missing values set to "Unknown" |
| `customer_type` | String | New or Returning, missing values set to "New" |
| `purchase` | Binary | 1 if the customer purchased, 0 if not (target) |

## Pipeline

### Task 1: Data cleaning
* Checks that `customer_id` is complete and unique, then casts it to integer.
* Fills missing values per column using the rules above.
* Casts every column to its required type.

### Task 2: Feature engineering
* Scales `time_spent`, `pages_viewed` and `basket_value` to the 0 to 1 range with `MinMaxScaler`.
* One hot encodes `device_type` and `customer_type` with `pd.get_dummies`, giving columns such as `device_type_Desktop` and `customer_type_New`.
* Keeps all other columns, so the feature set has 11 columns including `customer_id` and `purchase`.

### Task 3: Neural network
* Architecture: input layer, one hidden layer with 8 units and ReLU, output layer with 1 unit and Sigmoid.
* Loss: binary cross entropy (`BCELoss`).
* Optimizer: Adam, learning rate 0.01, 200 epochs.
* Validation: 20 percent of the labelled data held out with a stratified split (640 training rows, 160 holdout rows).
* Predictions: probabilities of 0.5 or higher are classified as a purchase. Results are stored in `validation_predictions` with the columns `customer_id` and `purchase`.

## Results

| Metric | Value |
|---|---|
| Training loss (epoch 50) | 0.4962 |
| Training loss (epoch 200) | 0.4738 |
| Holdout accuracy | 0.7875 |
| Holdout F1 | 0.8811 |

## Limitations and next steps

* Holdout accuracy is lower than the F1 score, which suggests purchases are the larger class and the model leans towards predicting a purchase. Check the class balance and compare against a majority class baseline.
* Report precision and recall for both classes, and try threshold tuning or class weights.
* The loss was still falling at epoch 200, so train longer or use mini batches with early stopping.
* Compare against tree based baselines such as Random Forest and XGBoost.

## Tech stack

Python, pandas, scikit learn, PyTorch

## Project structure

```
customer-purchase-prediction/
  README.md
  requirements.txt
  .gitignore
  customer_purchase_prediction.ipynb
  data/                  (local only, not committed)
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook customer_purchase_prediction.ipynb
```

Place `raw_customer_data.csv`, `model_data.csv`, `input_model_features.csv` and `validation_features.csv` in the `data/` folder (or next to the notebook) and run the cells in order. The datasets come from a case study exercise and are not included in this repository.

## Author

Happiness Chibuzor, AI Engineer
GitHub: [github.com/happinesschibuzor](https://github.com/happinesschibuzor)
