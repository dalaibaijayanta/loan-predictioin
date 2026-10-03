import pandas as pd


DATA_PATH = "data/cs-training.csv"
df = pd.read_csv(DATA_PATH)

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

print("=" * 70)
print("NORMAL VS SPECIAL DELINQUENCY VALUES")
print("=" * 70)


# ------------------------------------------------------------
# Create a flag for rows containing 96 or 98
# ------------------------------------------------------------

special_condition = (
    df[delinquency_columns]
    .isin([96, 98])
    .any(axis=1)
)

df["Special_Delinquency_Value"] = special_condition


# ------------------------------------------------------------
# Compare groups
# ------------------------------------------------------------

print("\nGroup sizes:")
print(
    df["Special_Delinquency_Value"]
    .value_counts()
)


print("\nDefault rate by group:")

default_rate = (
    df.groupby("Special_Delinquency_Value")[
        "SeriousDlqin2yrs"
    ]
    .mean()
    * 100
)

print(default_rate)


# ------------------------------------------------------------
# Compare delinquency values
# ------------------------------------------------------------

print("\nAverage delinquency values:")

print(
    df.groupby("Special_Delinquency_Value")[
        delinquency_columns
    ]
    .mean()
)


# ------------------------------------------------------------
# Compare age
# ------------------------------------------------------------

print("\nAverage age:")

print(
    df.groupby("Special_Delinquency_Value")["age"]
    .mean()
)


# ------------------------------------------------------------
# Compare debt ratio
# ------------------------------------------------------------

print("\nAverage DebtRatio:")

print(
    df.groupby("Special_Delinquency_Value")["DebtRatio"]
    .mean()
)


# ------------------------------------------------------------
# Compare revolving utilization
# ------------------------------------------------------------

print("\nAverage RevolvingUtilizationOfUnsecuredLines:")

print(
    df.groupby("Special_Delinquency_Value")[
        "RevolvingUtilizationOfUnsecuredLines"
    ]
    .mean()
)


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)