import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. LOAD DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "cs-training.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


# ============================================================
# 2. DISPLAY FIRST FEW ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ============================================================
# 3. COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

df.info()


# ============================================================
# 4. CHECK COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for column in df.columns:
    print(column)


# ============================================================
# 5. CHECK DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

missing_percentage = (missing_values / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

print(missing_table)


# ============================================================
# 7. TARGET DISTRIBUTION
# ============================================================

TARGET = "SeriousDlqin2yrs"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

target_counts = df[TARGET].value_counts()

target_percentage = df[TARGET].value_counts(normalize=True) * 100

target_table = pd.DataFrame({
    "Count": target_counts,
    "Percentage": target_percentage
})

print(target_table)


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL FEATURE STATISTICS")
print("=" * 60)

print(df.describe().T)


# ============================================================
# 9. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 60)
print("NUMBER OF UNIQUE VALUES")
print("=" * 60)

for column in df.columns:
    print(f"{column}: {df[column].nunique()}")


# ============================================================
# 10. TARGET VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x=TARGET
)

plt.title("Target Distribution")
plt.xlabel("Serious Delinquency Within 2 Years")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.show()


# ============================================================
# 11. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================================

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numeric_columns:

    if column == TARGET:
        continue

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=50
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of Applicants")

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. BOX PLOTS FOR OUTLIER INVESTIGATION
# ============================================================

for column in numeric_columns:

    if column == TARGET:
        continue

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df[column]
    )

    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. CORRELATION MATRIX
# ============================================================

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# 14. CORRELATION WITH TARGET
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

target_correlation = (
    correlation_matrix[TARGET]
    .sort_values(ascending=False)
)

print(target_correlation)


# ============================================================
# 15. TARGET VS IMPORTANT FEATURES
# ============================================================

important_features = [
    "age",
    "MonthlyIncome",
    "DebtRatio",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse"
]

for feature in important_features:

    if feature not in df.columns:
        continue

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=TARGET,
        y=feature
    )

    plt.title(f"{feature} vs Target")
    plt.xlabel("Serious Delinquency Within 2 Years")
    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()


print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)