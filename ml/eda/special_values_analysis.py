import pandas as pd


DATA_PATH = "data/cs-training.csv"
df = pd.read_csv(DATA_PATH)

columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

print("=" * 70)
print("SPECIAL VALUE ANALYSIS")
print("=" * 70)


for column in columns:

    print("\n" + "=" * 70)
    print(column)
    print("=" * 70)

    special_rows = df[df[column].isin([96, 98])]

    print(f"Number of rows containing 96 or 98: {len(special_rows)}")

    print("\nValue counts:")
    print(special_rows[column].value_counts().sort_index())

    print("\nTarget distribution:")
    print(
        special_rows["SeriousDlqin2yrs"]
        .value_counts()
    )

    print("\nDefault rate:")

    if len(special_rows) > 0:
        default_rate = (
            special_rows["SeriousDlqin2yrs"].mean() * 100
        )

        print(f"{default_rate:.2f}%")

    print("\nOther delinquency columns:")

    other_columns = [
        col for col in columns
        if col != column
    ]

    print(
        special_rows[
            other_columns + ["SeriousDlqin2yrs"]
        ].head(20)
    )


print("\n" + "=" * 70)
print("SPECIAL VALUE ANALYSIS COMPLETED")
print("=" * 70)