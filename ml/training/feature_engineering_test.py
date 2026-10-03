import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
)

from ml.preprocessing.cleaning import clean_raw_data


DATA_PATH = "data/cs-training.csv"
RANDOM_STATE = 42


# ---------------------------------------------------------
# Load raw data
# ---------------------------------------------------------

print("=" * 70)
print("LOADING RAW DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")


# ---------------------------------------------------------
# Create special-value indicator BEFORE cleaning
# ---------------------------------------------------------

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

df["SpecialDelinquencyFlag"] = (
    df[delinquency_columns]
    .isin([96, 98])
    .any(axis=1)
    .astype(int)
)

print(
    "\nSpecial delinquency rows detected:",
    df["SpecialDelinquencyFlag"].sum()
)


# ---------------------------------------------------------
# Clean data
# ---------------------------------------------------------

df = clean_raw_data(df)

print(f"Shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)

print(X.columns.tolist())


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


# ---------------------------------------------------------
# Model
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
# Train
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING MODEL WITH SPECIAL DELINQUENCY FLAG")
print("=" * 70)

model.fit(X_train, y_train)

print("Training completed successfully.")


# ---------------------------------------------------------
# Validation probabilities
# ---------------------------------------------------------

y_val_probability = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(
    y_val,
    y_val_probability
)


# ---------------------------------------------------------
# Test threshold = 0.20
# ---------------------------------------------------------

threshold = 0.20

y_val_pred = (
    y_val_probability >= threshold
).astype(int)


recall = recall_score(
    y_val,
    y_val_pred
)

precision = precision_score(
    y_val,
    y_val_pred
)

f1 = f1_score(
    y_val,
    y_val_pred
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Threshold : {threshold:.2f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ---------------------------------------------------------
# Comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PREVIOUS BEST RESULT")
print("=" * 70)

print("HistGradientBoosting without special flag:")
print("ROC-AUC   : 0.8670")
print("Recall    : 0.5279")
print("Precision : 0.3951")
print("F1 Score  : 0.4519")

print("\nThe test set was NOT used.")