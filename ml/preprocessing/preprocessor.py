import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def create_preprocessing_pipeline():
    """
    Create the preprocessing pipeline for the loan default dataset.

    Currently:
    - Missing numerical values are replaced with the median.

    The pipeline will be fitted only on training data.
    """

    preprocessing_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            )
        ]
    )

    return preprocessing_pipeline


if __name__ == "__main__":

    DATA_PATH = "data/cs-training.csv"
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")

    # Import our cleaning function
    from cleaning import clean_raw_data

    # Clean the raw data
    df = clean_raw_data(df)

    # Separate features and target
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    # ---------------------------------------------------------
    # Split data
    # ---------------------------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # ---------------------------------------------------------
    # Create preprocessing pipeline
    # ---------------------------------------------------------

    preprocessor = create_preprocessing_pipeline()

    # IMPORTANT:
    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the already-fitted pipeline to validation and test
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(f"Original training features : {X_train.shape}")
    print(f"Processed training features: {X_train_processed.shape}")

    print(f"Original validation features : {X_val.shape}")
    print(f"Processed validation features: {X_val_processed.shape}")

    print(f"Original test features : {X_test.shape}")
    print(f"Processed test features: {X_test_processed.shape}")

    print("\nMissing values before preprocessing:")
    print(X_train.isnull().sum())

    print("\nChecking processed training data...")

    import numpy as np

    print(
        f"Missing/NaN values after preprocessing: "
        f"{np.isnan(X_train_processed).sum()}"
    )

    print("\nPreprocessing pipeline:")
    print(preprocessor)