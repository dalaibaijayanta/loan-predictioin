import pandas as pd
from sklearn.model_selection import train_test_split

from cleaning import clean_raw_data


DATA_PATH = "data/cs-training.csv"


def main():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    # ---------------------------------------------------------
    # 1. Clean the data
    # ---------------------------------------------------------
    df = clean_raw_data(df)

    print(f"Shape after cleaning: {df.shape}")

    # ---------------------------------------------------------
    # 2. Separate features and target
    # ---------------------------------------------------------
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # ---------------------------------------------------------
    # 3. Create 80% training and 20% temporary data
    # ---------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # ---------------------------------------------------------
    # 4. Split temporary data into:
    #    10% validation
    #    10% test
    # ---------------------------------------------------------
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # ---------------------------------------------------------
    # 5. Print dataset sizes
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print(f"Training data   : {X_train.shape}")
    print(f"Validation data : {X_val.shape}")
    print(f"Test data       : {X_test.shape}")

    # ---------------------------------------------------------
    # 6. Check target distribution
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    print("\nTraining:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True))

    print("\nValidation:")
    print(y_val.value_counts())
    print(y_val.value_counts(normalize=True))

    print("\nTest:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True))


if __name__ == "__main__":
    main()