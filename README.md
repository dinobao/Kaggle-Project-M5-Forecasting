# Kaggle Project: M5 Forecasting
This project is a kaggle competition named ['M5 Forecasting'](https://www.kaggle.com/competitions/m5-forecasting-accuracy), completed for the **Artificial Intelligence in Fintech (MAFS5440)** course at **HKUST**.

## Project Objectives
- Process large-scale hierarchical dataset containing daily sales data, calendar events, and pricing information
- Apply LightGBM model to forecast daily sales across a 28-day horizon
- Combine recursive and non-recursive LightGBM models for accuracy prediction

## Data Prepocessing
  - Cleaned and merged firm-level and macroeconomic datasets (1957–2016)
  - Filled missing firm characteristics with cross-sectional medians and standardized features by period
  - Selected top and bottom 1000 firms by market value and created macro–firm interaction features for modeling.

## Recursive Prediction
In this project approach, we apply both recursive and non-recursive strategy to our forecasting task

<img src="https://raw.githubusercontent.com/dinobao/Kaggle-Project-M5-Forecasting/main/images/recursive.jpg" alt="Recursive Prediction" width="40%">

## Workflow
<img src="https://raw.githubusercontent.com/dinobao/Kaggle-Project-M5-Forecasting/main/images/accuracy.png" alt="Workflow" width="60%">


## Key Results
<img src="https://raw.githubusercontent.com/dinobao/Kaggle-Project-M5-Forecasting/main/images/result_accuracy.png" alt="Result" width="70%">

- In this project, we only use single LightGBM method in recursive and non-recursive way. The result shows that simple but practical model can produce very good results

## Files
- `code/1_feature_engineering`: Feature engineering codes
- `code/2_models`: First run recursive and non-recursive codes, then run `ensemble.py`
- `slides.pdf`: Final presentation slides

## Skills Demonstrated
Machine Learning · Feature Engineering · Hierarchical Data Analysis

## Connect
[Email](mailto:shijie_bao0209@outlook.com)

