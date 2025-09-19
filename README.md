# Cardiovascular_diseases-cardio-


## Overview

This repository contains a Jupyter notebook that explores, preprocesses, and models a cardiovascular disease dataset (binary classification: presence vs absence of cardiovascular disease). The notebook walks through data loading, exploratory data analysis (EDA), cleaning, feature engineering, model training (several tree- and linear-based models), evaluation and interpretation.

**Target:** `cardio` (binary label). The notebook assigns `X = df.drop('cardio', axis=1)` and `y = df['cardio']`.

## Dataset

The notebook loads the data from the original repository location:

```
https://raw.githubusercontent.com/Prachifox/Cardiovascular_diseases/master/cardio_train.csv
```

Typical columns in this dataset (as used in the notebook) include: `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc` (glucose), `smoke`, `alco`, `active`, and the target `cardio`.

## Notebook structure

The notebook is organized into the following sections (headings):

* Initial Imports and Data Loading
* Data Exploration
* Data Cleaning
* Feature Engineering
* Data Splitting
* Model Building
* Model Evaluation
* Model Interpretation

This structure makes it easy to follow the full ML pipeline from raw CSV to model interpretation.

## Models & Methods

The notebook trains and evaluates several models, including:

* LogisticRegression
* RandomForestClassifier
* XGBClassifier (XGBoost)
* LGBMClassifier (LightGBM)
* A StackingClassifier ensemble

Key preprocessing steps used in the notebook include:

* train/test split using `train_test_split`
* scaling features via `StandardScaler`
* basic cleaning and feature engineering (described in the notebook cells)

Evaluation metrics and analysis in the notebook include accuracy, precision, recall, F1-score, ROC AUC, ROC curve, and confusion matrix.

## How to run (quick)

1. Clone this repository:

```bash
git clone <this-repo-url>
cd <this-repo-folder>
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows
```

3. Install dependencies (suggested):

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the packages below manually.

4. Open the notebook and run all cells:

```bash
jupyter notebook Cardiovascular_diseases.ipynb
# then click "Run all" or run the notebook in JupyterLab/Colab
```

## Suggested requirements (`requirements.txt`)

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
jupyter
```

## Files in this repository

* `Cardiovascular_diseases.ipynb` — main Jupyter notebook (EDA, modeling, evaluation).
* (data) `cardio_train.csv` — dataset is loaded from the link provided in the notebook. You may include it locally in a `data/` folder if you prefer.

## Notes & Reproducibility

* The notebook uses common ML libraries (scikit-learn, XGBoost, LightGBM). For reproducible runs, set random seeds (e.g., `random_state` where appropriate) and pin package versions.
* If you plan to productionize a model, consider a more thorough pipeline with cross-validation, hyperparameter tuning, feature selection, and calibration.

## Results

Model evaluation figures and numeric results are presented inside the notebook under the **Model Evaluation** section (accuracy, precision, recall, F1, ROC AUC, confusion matrix). Please open the notebook to inspect numeric scores and plots for each model.
