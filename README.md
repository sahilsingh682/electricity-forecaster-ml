# Electricity Consumption Forecasting

## Live Application
🌍 **[Insert Your Streamlit Public URL Here]**

## Overview
This repository contains an end-to-end Machine Learning pipeline that predicts the next hour's household electricity consumption. 

## Methodology
1. **Data Processing:** Handled missing values via time-aware interpolation and resampled minute-level data to hourly intervals.
2. **Feature Engineering:** Extracted hour, day of the week, 1-hour lag, 24-hour lag, and 24-hour rolling mean.
3. **Feature Selection:** Standardized features using `StandardScaler` and applied Forward Feature Selection.
4. **Modeling:** Evaluated Ridge Regression, Lasso Regression, Principal Component Regression (PCR), and Partial Least Squares (PLS) utilizing `TimeSeriesSplit` cross-validation. 
5. **Evaluation:** Models were compared based on Mean Squared Error (MSE).

## Local Execution
To run this project locally:
1. `pip install -r requirements.txt`
2. `streamlit run app.py`