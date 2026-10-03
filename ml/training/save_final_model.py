import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "cs-training.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "loan_default_model.joblib"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

# Remove row identifier
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Treat age = 0 as missing
df.loc[df["age"] == 0, "age"] = pd.NA

# Treat 96 and 98 as special/anomalous values
# and convert them to missing values
for column in delinquency_columns:
    df.loc[df[column].isin([96, 98]), column] = pd.NA


# ---------------------------------------------------------
# FEATURES / TARGET
# ---------------------------------------------------------

TARGET = "SeriousDlqin2yrs"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


# ---------------------------------------------------------
# SAME TRAIN / VALIDATION / TEST SPLIT
# ---------------------------------------------------------

# First split:
# 80% training
# 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second split:
# 10% validation
# 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print()
print("Data split:")
print(f"Training:   {X_train.shape}")
print(f"Validation: {X_val.shape}")
print(f"Test:       {X_test.shape}")


# ---------------------------------------------------------
# FINAL MODEL
# ---------------------------------------------------------

model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "model",
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42
        )
    )
])


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

print()
print("Training final model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print()
print(f"Model saved successfully:")
print(MODEL_PATH)


# ---------------------------------------------------------
# SAVE THRESHOLD
# ---------------------------------------------------------

THRESHOLD = 0.21

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold.txt"
)

with open(threshold_path, "w") as file:
    file.write(str(THRESHOLD))

print(f"Threshold saved: {THRESHOLD}")


# ---------------------------------------------------------
# SAVE FEATURE NAMES
# ---------------------------------------------------------

feature_names_path = os.path.join(
    MODEL_DIR,
    "feature_names.txt"
)

with open(feature_names_path, "w") as file:
    for feature in X.columns:
        file.write(feature + "\n")

print(f"Feature names saved:")
print(feature_names_path)


# ---------------------------------------------------------
# FINAL INFORMATION
# ---------------------------------------------------------

print()
print("=" * 60)
print("FINAL MODEL PACKAGE CREATED")
print("=" * 60)

print(f"Model: HistGradientBoostingClassifier")
print(f"Threshold: {THRESHOLD}")
print(f"Training rows: {len(X_train)}")
print(f"Validation rows: {len(X_val)}")
print(f"Test rows: {len(X_test)}")

print()
print("Files created:")
print("1. models/loan_default_model.joblib")
print("2. models/threshold.txt")
print("3. models/feature_names.txt")

print("=" * 60)