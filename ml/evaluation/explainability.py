import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split


# =========================================================
# PROJECT PATHS
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cs-training.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "loan_default_model.joblib"
)


# =========================================================
# LOAD DATA
# =========================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# =========================================================
# CLEAN DATA
# =========================================================

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

# Remove identifier
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Age = 0 treated as missing
df.loc[df["age"] == 0, "age"] = pd.NA

# Special delinquency values treated as missing
for column in delinquency_columns:
    df.loc[df[column].isin([96, 98]), column] = pd.NA


# =========================================================
# FEATURES / TARGET
# =========================================================

TARGET = "SeriousDlqin2yrs"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =========================================================
# RECREATE ORIGINAL SPLIT
# =========================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

print("Loading saved model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# =========================================================
# GET PREPROCESSOR AND MODEL
# =========================================================

imputer = pipeline.named_steps["imputer"]
model = pipeline.named_steps["model"]


# =========================================================
# TRANSFORM VALIDATION DATA
# =========================================================

print("Transforming validation data...")

X_val_transformed = imputer.transform(X_val)


# =========================================================
# GET FEATURE NAMES
# =========================================================

original_feature_names = list(X.columns)

# The imputer adds missing-value indicator columns.
indicator_features = []

if hasattr(imputer, "indicator_") and imputer.add_indicator:

    indicator_indices = imputer.indicator_.features_

    for index in indicator_indices:
        indicator_features.append(
            f"{original_feature_names[index]}_missing"
        )

transformed_feature_names = (
    original_feature_names +
    indicator_features
)


print()
print(f"Original features: {len(original_feature_names)}")
print(f"Transformed features: {len(transformed_feature_names)}")


# =========================================================
# SHAP EXPLAINER
# =========================================================

print()
print("Creating SHAP explainer...")

explainer = shap.Explainer(
    model.predict_proba,
    X_val_transformed
)


# =========================================================
# EXPLAIN SAMPLE OF VALIDATION DATA
# =========================================================

# Use a small sample so explanation is reasonably fast.
sample_size = min(1000, len(X_val_transformed))

X_sample = X_val_transformed[:sample_size]

print(
    f"Generating SHAP explanations for "
    f"{sample_size} validation records..."
)

shap_values = explainer(X_sample)


# =========================================================
# HANDLE BINARY CLASSIFICATION
# =========================================================

# For binary classification with predict_proba,
# class 1 represents default.

if len(shap_values.values.shape) == 3:

    # Shape:
    # samples × features × classes

    default_shap_values = shap_values.values[:, :, 1]

else:

    default_shap_values = shap_values.values


# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

mean_abs_shap = np.abs(
    default_shap_values
).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": transformed_feature_names,
    "mean_abs_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "mean_abs_shap",
    ascending=False
)

print()
print("=" * 60)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance_df.head(10).to_string(index=False)
)


# =========================================================
# SAVE GLOBAL IMPORTANCE
# =========================================================

output_dir = os.path.join(
    PROJECT_ROOT,
    "models"
)

importance_path = os.path.join(
    output_dir,
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print()
print(
    f"Feature importance saved to:\n"
    f"{importance_path}"
)


# =========================================================
# LOCAL EXPLANATION
# =========================================================

sample_index = 0

local_values = default_shap_values[sample_index]

local_df = pd.DataFrame({
    "feature": transformed_feature_names,
    "shap_value": local_values,
    "absolute_shap": np.abs(local_values)
})

local_df = local_df.sort_values(
    "absolute_shap",
    ascending=False
)

print()
print("=" * 60)
print("LOCAL EXPLANATION")
print("=" * 60)

print(
    local_df.head(10).to_string(index=False)
)


# =========================================================
# PREDICTION FOR SAMPLE
# =========================================================

sample_probability = pipeline.predict_proba(
    X_val.iloc[[sample_index]]
)[0][1]

threshold = 0.21

prediction = (
    "HIGH RISK"
    if sample_probability >= threshold
    else "LOW RISK"
)

print()
print("=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

print(
    f"Default probability: "
    f"{sample_probability:.4f}"
)

print(
    f"Default probability: "
    f"{sample_probability * 100:.2f}%"
)

print(
    f"Decision threshold: "
    f"{threshold}"
)

print(
    f"Prediction: "
    f"{prediction}"
)

print("=" * 60)