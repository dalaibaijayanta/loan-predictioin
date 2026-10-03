import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

from ml.preprocessing.cleaning import clean_raw_data


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/cs-training.csv"
RANDOM_STATE = 42


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")

df = clean_raw_data(df)

print(f"Shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

print(f"\nFeatures: {X.shape}")
print(f"Target: {y.shape}")


# ---------------------------------------------------------
# Train / Validation / Test split
# ---------------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)


print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ---------------------------------------------------------
# HistGradientBoosting pipeline
# ---------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),

        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=RANDOM_STATE
            )
        )
    ]
)


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING HISTOGRAM GRADIENT BOOSTING")
print("=" * 70)

model.fit(X_train, y_train)

print("Training completed successfully.")


# ---------------------------------------------------------
# Validation predictions
# ---------------------------------------------------------

y_val_pred = model.predict(X_val)

y_val_probability = model.predict_proba(X_val)[:, 1]


# ---------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------

recall = recall_score(y_val, y_val_pred)

precision = precision_score(
    y_val,
    y_val_pred
)

f1 = f1_score(
    y_val,
    y_val_pred
)

roc_auc = roc_auc_score(
    y_val,
    y_val_probability
)


print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_matrix(
        y_val,
        y_val_pred
    )
)


# ---------------------------------------------------------
# Classification Report
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        y_val_pred,
        digits=4
    )
)


# ---------------------------------------------------------
# Protect test set
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET")
print("=" * 70)

print("Test set has NOT been used for model selection.")