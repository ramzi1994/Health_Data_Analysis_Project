# Health Data Analysis Mini Project

## Requirements
- Python 3.9 or newer
- NumPy
- Pandas
- Matplotlib

## Setup
Open a terminal in this folder and run:

```bash
pip install numpy pandas matplotlib
python health_data_analysis.py
```

If `health_data.csv` does not exist, the program creates a small synthetic sample dataset.
It cleans the data, performs NumPy filtering and summary calculations, then saves charts and a
cleaned CSV file in the `output` folder.

## Files created
- `health_data.csv` — sample dataset created on first run
- `output/cleaned_health_data.csv` — cleaned data plus age group
- `output/age_histogram.png`
- `output/gender_bar_chart.png`
- `output/age_vs_systolic_bp.png`
- `output/age_vs_cholesterol.png`
- `output/bp_vs_cholesterol.png`

## Dataset columns
`Patient_ID`, `Age`, `Gender`, `Systolic_BP`, `Diastolic_BP`, `Cholesterol`, `Glucose`

## Important
The included dataset is entirely synthetic for educational demonstration. It is not real patient data
and must not be used for diagnosis, treatment, or medical decision-making. Any thresholds shown are
simple filtering examples, not a diagnosis.
