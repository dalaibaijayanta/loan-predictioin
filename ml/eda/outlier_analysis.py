import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/cs-training.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("OUTLIER AND DATA VALIDITY ANALYSIS")
print("=" * 70)


# ============================================================
# 1. AGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. AGE ANALYSIS")
print("=" * 70)

print("Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())

print("\nApplicants younger than 18:")
print((df["age"] < 18).sum())

print("Applicants older than 100:")
print((df["age"] > 100).sum())

print("\nAge distribution:")
print(df["age"].describe())


# ============================================================
# 2. REVOLVING UTILIZATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("2. REVOLVING UTILIZATION ANALYSIS")
print("=" * 70)

column = "RevolvingUtilizationOfUnsecuredLines"

print(df[column].describe())

print("\nValues greater than 1:")
print((df[column] > 1).sum())

print("Values greater than 10:")
print((df[column] > 10).sum())

print("Values greater than 100:")
print((df[column] > 100).sum())

print("\nTop 10 largest values:")
print(df[column].nlargest(10).to_string(index=False))


# ============================================================
# 3. DEBT RATIO ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("3. DEBT RATIO ANALYSIS")
print("=" * 70)

column = "DebtRatio"

print(df[column].describe())

print("\nValues greater than 1:")
print((df[column] > 1).sum())

print("Values greater than 10:")
print((df[column] > 10).sum())

print("Values greater than 100:")
print((df[column] > 100).sum())

print("Values greater than 1000:")
print((df[column] > 1000).sum())

print("\nTop 10 largest values:")
print(df[column].nlargest(10).to_string(index=False))


# ============================================================
# 4. MONTHLY INCOME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("4. MONTHLY INCOME ANALYSIS")
print("=" * 70)

column = "MonthlyIncome"

print(df[column].describe())

print("\nValues greater than 50,000:")
print((df[column] > 50000).sum())

print("Values greater than 100,000:")
print((df[column] > 100000).sum())

print("Values greater than 1,000,000:")
print((df[column] > 1000000).sum())

print("\nTop 10 largest values:")
print(df[column].nlargest(10).to_string(index=False))


# ============================================================
# 5. DELINQUENCY COUNTS
# ============================================================

print("\n" + "=" * 70)
print("5. DELINQUENCY ANALYSIS")
print("=" * 70)

delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

for column in delinquency_columns:

    print("\n" + "-" * 50)
    print(column)
    print("-" * 50)

    print("Maximum:", df[column].max())

    print("Values greater than 10:")
    print((df[column] > 10).sum())

    print("Values greater than 20:")
    print((df[column] > 20).sum())

    print("Values greater than 50:")
    print((df[column] > 50).sum())

    print("\nValue counts:")
    print(df[column].value_counts().sort_index())


# ============================================================
# 6. OPEN CREDIT LINES
# ============================================================

print("\n" + "=" * 70)
print("6. OPEN CREDIT LINES ANALYSIS")
print("=" * 70)

column = "NumberOfOpenCreditLinesAndLoans"

print(df[column].describe())

print("\nValues greater than 30:")
print((df[column] > 30).sum())

print("Values greater than 50:")
print((df[column] > 50).sum())


# ============================================================
# 7. REAL ESTATE LOANS
# ============================================================

print("\n" + "=" * 70)
print("7. REAL ESTATE LOANS ANALYSIS")
print("=" * 70)

column = "NumberRealEstateLoansOrLines"

print(df[column].describe())

print("\nValues greater than 10:")
print((df[column] > 10).sum())

print("Values greater than 20:")
print((df[column] > 20).sum())


# ============================================================
# 8. DEPENDENTS
# ============================================================

print("\n" + "=" * 70)
print("8. DEPENDENTS ANALYSIS")
print("=" * 70)

column = "NumberOfDependents"

print(df[column].describe())

print("\nValues greater than 10:")
print((df[column] > 10).sum())

print("Values greater than 15:")
print((df[column] > 15).sum())

print("\nValue counts:")
print(df[column].value_counts(dropna=False).sort_index())


# ============================================================
# 9. SUSPICIOUS VALUES VS TARGET
# ============================================================

print("\n" + "=" * 70)
print("9. SUSPICIOUS VALUES VS TARGET")
print("=" * 70)


checks = {
    "Age > 100": df["age"] > 100,
    "Revolving Utilization > 10": df["RevolvingUtilizationOfUnsecuredLines"] > 10,
    "Debt Ratio > 100": df["DebtRatio"] > 100,
    "Monthly Income > 100000": df["MonthlyIncome"] > 100000,
    "90 Days Late > 20": df["NumberOfTimes90DaysLate"] > 20
}


for name, condition in checks.items():

    count = condition.sum()

    if count > 0:

        default_rate = df.loc[condition, "SeriousDlqin2yrs"].mean() * 100

        print(f"\n{name}")
        print(f"Number of applicants: {count}")
        print(f"Default rate: {default_rate:.2f}%")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER ANALYSIS COMPLETED")
print("=" * 70)