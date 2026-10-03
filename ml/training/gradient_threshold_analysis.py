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
# Load and clean data
# ---------------------------------------------------------

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")

df = clean_raw_data(df)

print(f"Shape after cleaning: {df.shape}")


# ---------------------------------------------------------
# Features and target
# ---------------------------------------------------------

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]


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
print("TRAINING HISTOGRAM GRADIENT BOOSTING")
print("=" * 70)

model.fit(X_train, y_train)

print("Training completed successfully.")


# ---------------------------------------------------------
# Probabilities
# ---------------------------------------------------------

y_val_probability = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(
    y_val,
    y_val_probability
)

print("\n" + "=" * 70)
print("ROC-AUC")
print("=" * 70)

print(f"ROC-AUC: {roc_auc:.4f}")


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

results = []


for threshold in thresholds:

    y_pred = (
        y_val_probability >= threshold
    ).astype(int)

    recall = recall_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    })


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Recall": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)


# ---------------------------------------------------------
# Best F1
# ---------------------------------------------------------

best_f1_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(f"Threshold : {best_f1_row['Threshold']:.2f}")
print(f"Recall    : {best_f1_row['Recall']:.4f}")
print(f"Precision : {best_f1_row['Precision']:.4f}")
print(f"F1        : {best_f1_row['F1']:.4f}")


print("\nTest set was NOT used.")