"""
Mini Project: Health Data Analysis
Tools: Python, NumPy, Pandas, Matplotlib
Dataset: health_data.csv (synthetic sample records)

IMPORTANT:
- This project uses synthetic/demo data, not real patient records.
- Do not use it for medical diagnosis or treatment.
- Install dependencies with: pip install numpy pandas matplotlib
- Run with: python health_data_analysis.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("health_data.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_sample_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Create a small synthetic dataset if no CSV is available."""
    if path.exists():
        return pd.read_csv(path)

    data = {
        "Patient_ID": np.arange(1001, 1021),
        "Age": [22, 35, 47, 56, 63, 29, 41, 52, 68, 31,
                45, 59, 72, 38, 50, 64, 27, 43, 55, 76],
        "Gender": ["Female", "Male", "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male",
                   "Female", "Male", "Female", "Male", "Female",
                   "Male", "Female", "Male", "Female", "Male"],
        "Systolic_BP": [112, 124, 131, 145, 152, 118, 128, 139, 158, 121,
                        135, 148, 162, 126, 137, 155, 115, 130, 143, 166],
        "Diastolic_BP": [72, 80, 84, 91, 94, 76, 82, 88, 96, 78,
                         86, 92, 98, 81, 87, 95, 74, 83, 90, 100],
        "Cholesterol": [165, 182, 198, 224, 236, 171, 190, 211, 248, 178,
                        205, 231, 255, 186, 215, 242, 169, 201, 226, 263],
        "Glucose": [88, 94, 101, 112, 119, 91, 98, 108, 126, 95,
                    104, 115, 132, 97, 109, 123, 89, 102, 114, 138],
    }
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and use NumPy to identify missing/invalid values."""
    df = df.copy()
    numeric_columns = ["Age", "Systolic_BP", "Diastolic_BP", "Cholesterol", "Glucose"]

    # Convert numeric columns safely; unparseable values become NaN.
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace impossible values with NaN before filling missing values.
    df.loc[(df["Age"] < 0) | (df["Age"] > 120), "Age"] = np.nan
    df.loc[(df["Systolic_BP"] <= 0) | (df["Systolic_BP"] > 300), "Systolic_BP"] = np.nan
    df.loc[(df["Diastolic_BP"] <= 0) | (df["Diastolic_BP"] > 200), "Diastolic_BP"] = np.nan
    df.loc[(df["Cholesterol"] <= 0) | (df["Cholesterol"] > 1000), "Cholesterol"] = np.nan
    df.loc[(df["Glucose"] <= 0) | (df["Glucose"] > 1000), "Glucose"] = np.nan

    # Use NumPy's finite checks and median to impute missing numeric values.
    for column in numeric_columns:
        values = df[column].to_numpy(dtype=float)
        valid_values = values[np.isfinite(values)]
        if valid_values.size:
            median_value = np.median(valid_values)
            df[column] = np.where(np.isfinite(values), values, median_value)

    # Remove duplicate rows.
    df = df.drop_duplicates()
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add age categories for demographic visualization."""
    df = df.copy()
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 17, 29, 44, 59, 120],
        labels=["0-17", "18-29", "30-44", "45-59", "60+"],
        include_lowest=True,
    )
    return df


def create_visualizations(df: pd.DataFrame) -> None:
    """Save demographic charts and relationship scatter plots."""
    # 1. Histogram: patient age distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["Age"], bins=8, edgecolor="black")
    plt.title("Patient Age Distribution")
    plt.xlabel("Age (years)")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_histogram.png", dpi=150)
    plt.close()

    # 2. Bar plot: gender counts
    gender_counts = df["Gender"].fillna("Unknown").value_counts()
    plt.figure(figsize=(7, 5))
    plt.bar(gender_counts.index.astype(str), gender_counts.values, edgecolor="black")
    plt.title("Patient Demographics by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Patients")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gender_bar_chart.png", dpi=150)
    plt.close()

    # 3. Scatter plot: age vs systolic blood pressure
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Age"], df["Systolic_BP"])
    plt.title("Age vs Systolic Blood Pressure")
    plt.xlabel("Age (years)")
    plt.ylabel("Systolic Blood Pressure (mmHg)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_vs_systolic_bp.png", dpi=150)
    plt.close()

    # 4. Scatter plot: age vs cholesterol
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Age"], df["Cholesterol"])
    plt.title("Age vs Cholesterol")
    plt.xlabel("Age (years)")
    plt.ylabel("Cholesterol (mg/dL)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "age_vs_cholesterol.png", dpi=150)
    plt.close()

    # 5. Scatter plot: systolic BP vs cholesterol
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Systolic_BP"], df["Cholesterol"])
    plt.title("Systolic Blood Pressure vs Cholesterol")
    plt.xlabel("Systolic Blood Pressure (mmHg)")
    plt.ylabel("Cholesterol (mg/dL)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bp_vs_cholesterol.png", dpi=150)
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics and example NumPy filtering."""
    print("\n=== HEALTH DATA ANALYSIS: SUMMARY ===")
    print(f"Number of patient records: {len(df)}")
    print("\nDescriptive statistics:")
    print(df[["Age", "Systolic_BP", "Diastolic_BP", "Cholesterol", "Glucose"]].describe().round(2))

    ages = df["Age"].to_numpy()
    systolic_bp = df["Systolic_BP"].to_numpy()
    cholesterol = df["Cholesterol"].to_numpy()

    # NumPy boolean masks demonstrate filtering operations.
    older_patients = ages >= 60
    higher_systolic = systolic_bp >= 140
    higher_cholesterol = cholesterol >= 200

    print(f"\nPatients aged 60 or above: {int(np.sum(older_patients))}")
    print(f"Records with systolic BP >= 140 mmHg: {int(np.sum(higher_systolic))}")
    print(f"Records with cholesterol >= 200 mg/dL: {int(np.sum(higher_cholesterol))}")

    # Correlation is descriptive only and does not imply causation.
    correlation = np.corrcoef(ages, systolic_bp)[0, 1]
    print(f"Pearson correlation (Age, Systolic BP): {correlation:.3f}")
    print("\nNote: These are synthetic demonstration records, not clinical findings.")


def main() -> None:
    print("Loading dataset...")
    raw_df = create_sample_dataset()
    print(f"Loaded {len(raw_df)} rows.")

    cleaned_df = clean_data(raw_df)
    cleaned_df = add_age_group(cleaned_df)

    cleaned_df.to_csv(OUTPUT_DIR / "cleaned_health_data.csv", index=False)
    create_visualizations(cleaned_df)
    print_summary(cleaned_df)

    print(f"\nDone. Cleaned data and charts are saved in: {OUTPUT_DIR.resolve()}")
    print("Charts: age_histogram.png, gender_bar_chart.png, age_vs_systolic_bp.png,")
    print("        age_vs_cholesterol.png, bp_vs_cholesterol.png")


if __name__ == "__main__":
    main()
