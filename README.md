# Predictive Modeling using Random Forest in Python

This repository contains a Python-based machine learning project that builds and evaluates a **Predictive Model** using **Random Forest Regressor**. The project allows users to load datasets in CSV, Excel, or JSON formats, preprocess the data, and train a Random Forest model. It also includes functionality for hyperparameter tuning using **GridSearchCV** to optimize model performance.

## Features:
- **Data Loading**: Supports data input from CSV, Excel, or JSON files.
- **Preprocessing**: 
  - Missing values handled using the median imputation method.
  - Categorical variables transformed into numerical values using one-hot encoding.
  - Data scaling with `StandardScaler` for consistent feature scaling.
- **Model Training**: 
  - Utilizes the **Random Forest Regressor** for predictive modeling.
  - Hyperparameter tuning with **GridSearchCV** to find the optimal model parameters.
- **Evaluation**: 
  - Evaluation metrics include **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R² Score**.
  - Visualizations include:
    - Feature importance bar plots
    - Error distribution histograms
- **Model Saving**: Save and load trained models using the **Joblib** library for future predictions.

## Libraries Used:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `joblib`

## How to Use:
1. Clone the repository and install the required libraries.
2. Run the script and choose the appropriate dataset file (CSV, Excel, or JSON).
3. Select the target variable and features for model training.
4. View the evaluation metrics and visualizations to assess model performance.
5. Save the trained model for future predictions.

## License:
This project is licensed under the MIT License.
