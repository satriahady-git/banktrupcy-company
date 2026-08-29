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
9. Save the selected Random Forest model and its metadata.

After preprocessing, the model uses 83 financial features.

## Model Results

The following results were produced using a classification threshold of 0.50:

| Metric | Random Forest | XGBoost | LightGBM |
|---|---:|---:|---:|
| Training PR-AUC | 0.841 | 0.587 | 0.575 |
| CV PR-AUC | 0.417 | **0.420** | 0.397 |
| Test PR-AUC | **0.458** | 0.441 | 0.438 |
| Test ROC-AUC | **0.939** | 0.932 | 0.933 |
| Precision | **0.455** | 0.234 | 0.258 |
| Recall | 0.686 | **0.784** | 0.765 |
| F1-score | **0.547** | 0.360 | 0.386 |
| F2-score | **0.623** | 0.533 | 0.549 |
| MCC | **0.538** | 0.392 | 0.411 |
| False positives | **42** | 131 | 112 |
| False negatives | 16 | **11** | 12 |

Random Forest was selected as the final model because it achieved the best test PR-AUC, ROC-AUC, precision, F1-score, F2-score, and MCC. It also produced far fewer false positives than XGBoost and LightGBM.

XGBoost achieved the highest recall, but it generated many more false-positive predictions. It may be preferred when missing a bankrupt company is much more expensive than investigating a false alarm.

## Final Random Forest Parameters

The selected Random Forest uses:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=10,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=1,
)
```

## Project Structure

```text
default_prediction/
|-- artifacts/
|   |-- random_forest_model.joblib
|   `-- random_forest_metadata.json
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

Install the saved requirements and the main modeling libraries:

```powershell
python -m pip install -r requirements.txt
python -m pip install scikit-learn==1.7.2 xgboost==3.0.5 matplotlib==3.10.9 seaborn==0.13.2 kagglehub shap joblib
```

Start Jupyter and open the notebook:

```powershell
jupyter notebook modeling.ipynb
```

Run the cells from top to bottom to reproduce the data analysis, model training, and evaluation.

## Load the Saved Model

The final fitted Random Forest pipeline is stored in `artifacts/random_forest_model.joblib`. Its parameters, feature names, threshold, and test metrics are stored in `artifacts/random_forest_metadata.json`.

```python
import json
import joblib

from pathlib import Path

artifact_dir = Path("artifacts")

model = joblib.load(
    artifact_dir / "random_forest_model.joblib"
)

with (
    artifact_dir / "random_forest_metadata.json"
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

The saved pipeline contains the fitted Random Forest but does not automatically perform the manual data cleaning from the notebook. New raw data must first receive the same preprocessing:

- Remove the same 12 mixed-scale columns.
- Replace values greater than or equal to 1,000,000 in the selected ratio columns with missing values.
- Keep the exact feature names and column order stored in the metadata file.

Do not evaluate the model on rows that were used during training. Model performance should be measured using unseen test data.

## Limitations

- The dataset is strongly imbalanced, so accuracy alone can be misleading.
- Random Forest has a noticeable difference between training and cross-validation PR-AUC.
- The 0.50 classification threshold may not be optimal for every business case.
- Threshold tuning should use validation data rather than the final test set.
- The model is an educational example and should not be used as the only basis for a real financial decision.

## License

See [`LICENSE`](LICENSE) for the project license.
