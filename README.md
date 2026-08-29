# Company Bankruptcy Prediction

This is a beginner machine learning project for predicting whether a company may go bankrupt from financial ratio data.

The project compares three tree-based classification models:

- Random Forest
- XGBoost
- LightGBM

The complete analysis is available in [`modeling.ipynb`](modeling.ipynb).

## Dataset

The project uses the [Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction) dataset from Kaggle.

The original dataset contains:

- 6,819 companies
- 95 financial features
- 1 target column named `Bankrupt?`
- 220 bankrupt companies
- 6,599 non-bankrupt companies

Only about 3.2% of the companies are bankrupt. Because the target is highly imbalanced, PR-AUC is used as the main model evaluation metric.

## Project Workflow

The notebook follows these steps:

1. Download and load the dataset.
2. Explore its shape, missing values, target distribution, and feature distributions.
3. Examine correlations between the financial features and bankruptcy.
4. Remove 12 mixed-scale columns with unusual values.
5. Replace invalid ratio values greater than or equal to 1,000,000 with missing values.
6. Split the processed data into 80% training data and 20% test data.
7. Tune each model using randomized search and 5-fold stratified cross-validation.
8. Compare the models on the same test set.
9. Save the selected XGBoost model and its metadata.

After preprocessing, the model uses 83 financial features.

## Model Results

The following results were produced using a classification threshold of 0.50:

| Metric | Random Forest | XGBoost | LightGBM |
|---|---:|---:|---:|
| Training PR-AUC | 0.960 | **0.998** | 0.921 |
| CV PR-AUC | 0.413 | 0.402 | **0.414** |
| CV standard deviation | 0.090 | **0.047** | 0.064 |
| Test PR-AUC | **0.492** | 0.477 | 0.475 |
| Test ROC-AUC | **0.953** | **0.953** | 0.947 |
| Precision | **0.431** | 0.412 | 0.321 |
| Recall | 0.500 | 0.750 | **0.773** |
| F1-score | 0.463 | **0.532** | 0.453 |
| F2-score | 0.485 | **0.645** | 0.603 |
| MCC | 0.445 | **0.537** | 0.474 |
| False positives | **29** | 47 | 72 |
| False negatives | 22 | 11 | **10** |

The cross-validation PR-AUC scores were close relative to their fold-to-fold variation, so there was no decisive validation winner. The final choice therefore also considers the operational goal of detecting bankrupt companies while maintaining reasonable precision.

**XGBoost was selected as the final model.** Its test PR-AUC was only 0.015 below Random Forest, while its recall improved from 0.500 to 0.750. XGBoost also achieved the highest F1-score, F2-score, and MCC. LightGBM provided slightly higher recall, but XGBoost had substantially better precision and generated 25 fewer false positives.

This selection prioritizes bankruptcy detection. The classification threshold should ultimately be selected from out-of-fold validation predictions using a defined business cost or recall requirement, without using the test set for threshold tuning.

## Final XGBoost Parameters

The selected XGBoost model uses:

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    subsample=0.6,
    colsample_bytree=0.6,
    reg_alpha=5,
    reg_lambda=5,
    scale_pos_weight=29.994,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)
```

## Project Structure

```text
default_prediction/
|-- artifacts/
|   |-- xgboost_model.joblib
|   `-- xgboost_metadata.json
|-- data/
|   `-- raw/company_bankruptcy/
|       |-- data.csv
|       `-- data_model.csv
|-- config.py
|-- modeling.ipynb
|-- requirements.txt
`-- README.md
```

`data.csv` is the original dataset. `data_model.csv` is the processed dataset used for model training.

## Setup

Python 3.11 is recommended because that is the version used to save and test the final model.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start Jupyter and open the notebook:

```powershell
jupyter notebook modeling.ipynb
```

Run the cells from top to bottom to reproduce the data analysis, model training, and evaluation.

## Export and Load the Final Model

Run the notebook through its final export cell to generate `artifacts/xgboost_model.joblib` and `artifacts/xgboost_metadata.json`. The metadata contains the fitted parameters, feature names, classification threshold, library version, and test metrics.

```python
import json
import joblib

from pathlib import Path

artifact_dir = Path("artifacts")

model = joblib.load(
    artifact_dir / "xgboost_model.joblib"
)

with (
    artifact_dir / "xgboost_metadata.json"
).open("r", encoding="utf-8") as file:
    metadata = json.load(file)

feature_names = metadata["feature_names"]
threshold = metadata["classification_threshold"]

# The dataframe must already have the same preprocessing
# and features used during model training.
X_new = df_new.loc[:, feature_names].copy()

probability = model.predict_proba(X_new)[:, 1]
prediction = (probability >= threshold).astype(int)

df_predictions = df_new.copy()
df_predictions["bankruptcy_probability"] = probability
df_predictions["bankruptcy_prediction"] = prediction
```

## Important Prediction Note

The saved pipeline contains the fitted XGBoost classifier but does not automatically perform the manual data cleaning from the notebook. New raw data must first receive the same preprocessing:

- Remove the same 12 mixed-scale columns.
- Replace values greater than or equal to 1,000,000 in the selected ratio columns with missing values.
- Keep the exact feature names and column order stored in the metadata file.

Do not evaluate the model on rows that were used during training. Model performance should be measured using unseen test data.

## Limitations

- The dataset is strongly imbalanced, so accuracy alone can be misleading.
- All three models have noticeable differences between training and cross-validation PR-AUC, indicating overfitting risk.
- Cross-validation did not identify a decisive winner, so the XGBoost selection also reflects the recall-oriented project objective.
- The 0.50 classification threshold may not be optimal for every business case.
- Threshold tuning should use validation data rather than the final test set.
- The model is an educational example and should not be used as the only basis for a real financial decision.
