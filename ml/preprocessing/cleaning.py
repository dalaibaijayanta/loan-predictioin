import pandas as pd


# Columns containing delinquency information
delinquency_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]


def clean_raw_data(df):
    """
    Perform basic data cleaning on the Give Me Some Credit dataset.

    This function:
    1. Removes the row identifier.
    2. Converts invalid age values to missing.
    3. Converts special delinquency values (96 and 98) to missing.
    """

    df = df.copy()

    # Remove row identifier
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Age = 0 is invalid, so treat it as missing
    df.loc[df["age"] == 0, "age"] = pd.NA

    # Convert special delinquency values 96 and 98 to missing
    for column in delinquency_columns:
        df.loc[df[column].isin([96, 98]), column] = pd.NA

    return df


if __name__ == "__main__":

  
    DATA_PATH = "data/cs-training.csv"
    print("=" * 70)
    print("LOADING RAW DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    # Clean the data
    cleaned_df = clean_raw_data(df)

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)

    print(f"Cleaned shape: {cleaned_df.shape}")

    print("\nColumns after cleaning:")
    print(cleaned_df.columns.tolist())

    print("\nMissing values after cleaning:")

    missing = cleaned_df.isnull().sum()

    print(
        missing[missing > 0]
        .sort_values(ascending=False)
    )

    print("\nSpecial 96/98 values remaining:")

    for column in delinquency_columns:
        count = cleaned_df[column].isin([96, 98]).sum()
        print(f"{column}: {count}")

    print("\nInvalid age values remaining:")
    print((cleaned_df["age"] == 0).sum())