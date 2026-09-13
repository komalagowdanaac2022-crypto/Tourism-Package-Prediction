
import os
import pandas as pd


def validate_dataset(file_path, expected_columns):
    """
    Validate the tourism dataset by checking:
    1. File existence
    2. Expected columns
    3. Missing and extra columns
    4. Dataset shape
    5. Missing values
    6. Dataset summary
    """

    print("=" * 70)
    print("DATA REGISTRATION AND VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Check whether dataset exists
    # ---------------------------------------------------------
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset not found at: {file_path}")
        return False

    # ---------------------------------------------------------
    # 2. Load dataset
    # ---------------------------------------------------------
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")
    print(f"File path: {file_path}")

    # ---------------------------------------------------------
    # 3. Check expected columns
    # ---------------------------------------------------------
    actual_columns = list(df.columns)

    missing_columns = [
        column for column in expected_columns
        if column not in actual_columns
    ]

    extra_columns = [
        column for column in actual_columns
        if column not in expected_columns
    ]

    print("\n" + "-" * 70)
    print("COLUMN VALIDATION")
    print("-" * 70)

    print(f"Expected number of columns: {len(expected_columns)}")
    print(f"Actual number of columns:   {len(actual_columns)}")

    if missing_columns:
        print("\nMissing columns:")
        for column in missing_columns:
            print(f"  - {column}")
    else:
        print("\nNo expected columns are missing.")

    if extra_columns:
        print("\nExtra columns:")
        for column in extra_columns:
            print(f"  - {column}")
    else:
        print("No extra columns found.")

    # ---------------------------------------------------------
    # 4. Dataset information
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("DATASET SUMMARY")
    print("-" * 70)

    print(f"Number of rows:    {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"  - {column}")

    # ---------------------------------------------------------
    # 5. Missing value summary
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing_values = df.isnull().sum()

    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(missing_values[missing_values > 0])

    # ---------------------------------------------------------
    # 6. Data types
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(df.dtypes)

    # ---------------------------------------------------------
    # 7. Statistical summary
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("STATISTICAL SUMMARY")
    print("-" * 70)

    print(df.describe(include="all").transpose())

    # ---------------------------------------------------------
    # 8. Final validation result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)

    if not missing_columns and not extra_columns:
        print("DATASET VALIDATION PASSED")
        print("=" * 70)
        return True
    else:
        print("DATASET VALIDATION FAILED")
        print("=" * 70)
        return False


# =============================================================
# Main program
# =============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        "/content/Tourism-Package-Prediction/data/tourism.csv"
    )

    EXPECTED_COLUMNS = [
        "Unnamed: 0",
        "CustomerID",
        "ProdTaken",
        "Age",
        "TypeofContact",
        "CityTier",
        "DurationOfPitch",
        "Occupation",
        "Gender",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "ProductPitched",
        "PreferredPropertyStar",
        "MaritalStatus",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "Designation",
        "MonthlyIncome"
    ]

    validate_dataset(
        DATASET_PATH,
        EXPECTED_COLUMNS
    )
