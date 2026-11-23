# Sri Lankan House Price Prediction Using Ensemble Learning

## 1. Project Overview

This project implements and compares different Ensemble Machine Learning techniques for predicting residential property asking prices in Sri Lanka.

The project investigates four ensemble learning approaches:

- **Bagging:** Random Forest Regressor
- **Boosting:** XGBoost and CatBoost
- **Stacking:** Random Forest, XGBoost, and CatBoost combined using a Ridge Regression meta-learner
- **Hybridization:** Equal-weight combinations of multiple trained models

An interactive dashboard was developed using Streamlit to visualize the dataset, compare model performance, and generate house-price predictions.

**Objective:** Evaluate how different ensemble learning approaches perform on Sri Lankan property advertisement data and determine whether combining multiple models improves prediction accuracy.

## 2. Dataset

**Dataset:** Sri Lanka Property Ads Dataset

**Source:** [Kaggle — Sri Lanka Property Ads Dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset)

The original dataset includes different categories of property advertisements in Sri Lanka. For this project, only advertisements categorized as *Houses For Sale* were considered.

Following data cleaning, filtering, and likely duplicate-advertisement removal, the final dataset contained **33,337 house listings**.

### Input Features

| Feature | Description |
|---|---|
| `location` | Location of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `house_size_sqft` | House size in square feet |
| `land_size_perches` | Land area in perches |

**Target variable:** `price_million` — advertised house price expressed in millions of Sri Lankan Rupees.

### Dataset Splitting

| Subset | Records | Purpose |
|---|---:|---|
| Training | 21,222 | Training machine learning models |
| Validation | 5,372 | Model comparison and selection |
| Testing | 6,743 | Final evaluation |

Group-aware splitting was used to prevent identical input-feature combinations from appearing across the training, validation, and testing subsets.

**Important:** The data contains historical advertised asking prices, not confirmed property transaction prices. Predictions should not be interpreted as current market valuations.

## 3. Machine Learning Models

### 3.1 Baseline Model

A `DummyRegressor` using the median training price was implemented as a reference model.

### 3.2 Random Forest — Bagging

Random Forest combines predictions from multiple Decision Trees trained using bootstrap samples and random feature selection.

Configuration:

- Number of trees: 100
- Maximum depth: 18
- Minimum samples per leaf: 3
- Random state: 42

### 3.3 XGBoost — Boosting

XGBoost uses gradient boosting to sequentially improve an ensemble of Decision Trees.

Configuration:

- Number of estimators: 200
- Maximum depth: 6
- Learning rate: 0.05
- Subsample ratio: 0.8
- Feature subsample ratio: 0.8

### 3.4 CatBoost — Boosting

CatBoost is a gradient-boosting algorithm that provides native support for categorical features.

Configuration:

- Iterations: 350
- Tree depth: 6
- Learning rate: 0.05
- Loss function: RMSE
- Categorical feature: Location

### 3.5 Stacking Ensemble

The Stacking Ensemble combines three trained base models:

1. Random Forest
2. XGBoost
3. CatBoost

A **Ridge Regression** model with `alpha = 10.0` was used as the meta-learner.

Three-fold `GroupKFold` cross-validation was used to generate out-of-fold predictions for training the meta-learner.

Stacking was selected as the final model based on validation MAE.

### 3.6 Hybrid Ensemble Models

Two fixed-weight ensemble approaches were evaluated:

**Hybrid RF + XGBoost**

The final prediction is the average of Random Forest and XGBoost predictions.

**Equal-Weight Ensemble**

The final prediction is the average of Random Forest, XGBoost, and CatBoost predictions.

Unlike Stacking, these combinations do not require training an additional meta-learner.

## 4. Model Evaluation Results

Models were evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

MAE and RMSE are reported in millions of Sri Lankan Rupees.

### Validation Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Stacking | **10.6714** | **27.3217** | **0.6400** |
| Equal-Weight Ensemble | 10.6773 | 27.8385 | 0.6263 |
| Hybrid RF + XGBoost | 10.9111 | 28.5140 | 0.6079 |
| XGBoost | 10.9345 | 27.4908 | 0.6355 |
| CatBoost | 11.0137 | 28.3088 | 0.6135 |
| Random Forest | 11.2937 | 30.7124 | 0.5451 |
| Baseline | 22.6346 | 46.9156 | -0.0615 |

Stacking achieved the lowest validation MAE and was therefore selected for final evaluation.

### Final Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Equal-Weight Ensemble | **9.9123** | 23.2264 | 0.7117 |
| Stacking | 9.9309 | **22.9200** | **0.7192** |
| Hybrid RF + XGBoost | 10.1125 | 23.5192 | 0.7044 |
| XGBoost | 10.2573 | 23.1249 | 0.7142 |
| CatBoost | 10.3163 | 24.5849 | 0.6770 |
| Random Forest | 10.3648 | 25.0012 | 0.6659 |
| Baseline | 21.7358 | 44.4029 | -0.0538 |

### Key Findings

- All trained machine learning models outperformed the baseline.
- XGBoost achieved the lowest test MAE among the individual models.
- Stacking achieved the lowest validation MAE.
- Equal-Weight Averaging achieved a slightly lower test MAE than Stacking.
- Stacking achieved the best test RMSE and R².
- The performance improvements between the strongest ensembles were relatively small.

Although Equal-Weight Averaging achieved the best test MAE, Stacking remained the selected model because selection had already been performed using validation results.

## 5. Interactive Dashboard

The Streamlit dashboard contains four main sections.

**Dataset Overview**

Displays dataset statistics, house-price distributions, location frequencies, and a dataset preview.

**Model Comparison**

Displays validation and test results for seven prediction approaches, including interactive comparison charts.

**House Price Prediction**

Allows users to enter property details and predict advertised house prices using a selected model. Predictions from all models can also be compared.

**Model Insights**

Explains Bagging, Boosting, Stacking, and Hybridization, including the learned Ridge meta-learner coefficients.

## 6. Project Structure

```text
ensemble_dashboard/
│
├── notebooks/
│   └── Sri_Lanka_Ensemble_Learning.ipynb
│
├── data/
│   ├── housing_clean.csv
│   ├── validation_results.csv
│   └── test_results.csv
│
├── models/
│   ├── baseline.joblib
│   ├── random_forest.joblib
│   ├── catboost.joblib
│   ├── stacking_meta.joblib
│   ├── xgboost_native.json
│   └── xgboost_preprocessor.joblib
│
├── app.py
├── metadata.json
├── requirements.txt
└── README.md
```

Additional files used during XGBoost portability verification may also be present inside `models/`.

## 7. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- CatBoost
- SciPy
- Joblib
- Streamlit
- Plotly
- Google Colab
- Visual Studio Code

## 8. Installation and Execution

### Prerequisites

- Python 3.13
- Internet connection for installing Python packages
- Project files, including the trained models and cleaned dataset

The dashboard was successfully tested using Python 3.13 on Windows.

### Step 1 — Open the Project

Open the `ensemble_dashboard` folder in Visual Studio Code.

### Step 2 — Create a Virtual Environment

In PowerShell, run:

```powershell
py -3.13 -m venv .venv
```

### Step 3 — Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The compatible SciPy version used during local testing was `1.16.3`. Ensure the requirements file includes this version.

### Step 4 — Start the Dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Step 5 — Open the Application

Open the following URL in a web browser:

http://localhost:8501

The dashboard loads the previously trained models and allows users to explore the results and generate predictions without retraining.

## 9. Reproducing the Training Experiment

The training notebook is available in:

`notebooks/Sri_Lanka_Ensemble_Learning.ipynb`

To reproduce the training workflow:

1. Open the notebook in Google Colab.
2. Download the original `properties.csv` dataset from Kaggle.
3. Upload `properties.csv` into the Colab runtime at `/content/properties.csv`.
4. Install any missing dependencies.
5. Execute the notebook cells in their original order.
6. Review the preprocessing, training, validation, and testing results.

The original dataset is not the same as the cleaned dataset provided for the dashboard.

The exported XGBoost model uses its native JSON format, along with a separately saved preprocessing pipeline, for portability between Google Colab and the local environment.

## 10. Limitations

- The data represents historical advertised asking prices rather than verified sale prices.
- Only five property characteristics are used as model inputs.
- Property condition, accessibility, and changing market conditions are not explicitly modeled.
- Some repeated property advertisements may remain.
- Outlier filtering may exclude unusual but valid properties.
- Prediction reliability may vary for locations with limited training examples.
- Extensive hyperparameter optimization was not performed.

The system is intended for educational and experimental purposes rather than professional property valuation.

## 11. Conclusion

This project demonstrates the application of Bagging, Boosting, Stacking, and Hybrid Ensemble approaches to a real Sri Lankan house-price prediction problem.

The experiments show that combining multiple trained models can improve predictive performance. Stacking achieved the best validation MAE and the best final test RMSE and R², while Equal-Weight Averaging remained highly competitive.

The Streamlit dashboard provides an interactive way to explore the dataset, compare model performance, and generate house-price predictions.

## 12. AI Assistance Acknowledgment

ChatGPT was used to support code development, debugging, explanation of machine learning concepts, and documentation. Experimental results were obtained by executing the machine learning workflow on the dataset.

## 13. Dataset Reference

Ivantha. *Sri Lanka Property Ads Dataset*. Kaggle.

https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset