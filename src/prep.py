
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = "/content/Tourism-Package-Prediction"

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "tourism.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "prepared_data"
)

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)


# ============================================================
# DATA PREPARATION FUNCTION
# ============================================================

def prepare_data():

    print("=" * 70)
    print("DATA PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset:")
    print(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    print("\nOriginal dataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # 3. Remove unnecessary column
    # --------------------------------------------------------

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("\nRemoved unnecessary column: Unnamed: 0")

    # --------------------------------------------------------
    # 4. Remove duplicate rows
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows found: {duplicates}")

    if duplicates > 0:
        df = df.drop_duplicates()

    print("Shape after removing duplicates:")
    print(df.shape)

    # --------------------------------------------------------
    # 5. Handle missing numeric values
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    print("\nHandling missing numeric values...")

    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

            print(
                f"Filled missing values in {column} "
                f"with median: {median_value}"
            )

    # --------------------------------------------------------
    # 6. Handle missing categorical values
    # --------------------------------------------------------

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    print("\nHandling missing categorical values...")

    for column in categorical_columns:
        if df[column].isnull().sum() > 0:
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)

            print(
                f"Filled missing values in {column} "
                f"with mode: {mode_value}"
            )

    # --------------------------------------------------------
    # 7. Check remaining missing values
    # --------------------------------------------------------

    remaining_missing = df.isnull().sum().sum()

    print("\nRemaining missing values:", remaining_missing)

    # --------------------------------------------------------
    # 8. Train-test split
    # --------------------------------------------------------

    print("\nSplitting dataset into train and test sets...")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["ProdTaken"]
    )

    print("\nTraining data shape:")
    print(train_df.shape)

    print("Testing data shape:")
    print(test_df.shape)

    # --------------------------------------------------------
    # 9. Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 10. Save train and test datasets
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nTraining data saved to:")
    print(TRAIN_PATH)

    print("\nTesting data saved to:")
    print(TEST_PATH)

    # --------------------------------------------------------
    # 11. Final information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nPrepared data folder:")
    print(OUTPUT_DIR)

    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prepare_data()
