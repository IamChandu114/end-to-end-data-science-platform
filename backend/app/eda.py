import pandas as pd


def automated_eda(df: pd.DataFrame):
    eda_report = {}

    # Shape
    eda_report["shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    # Column types
    eda_report["column_types"] = df.dtypes.astype(str).to_dict()

    # Missing values
    eda_report["missing_values"] = df.isnull().sum().to_dict()

    # Numeric & Categorical columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    eda_report["numeric_columns"] = numeric_cols
    eda_report["categorical_columns"] = categorical_cols

    # Basic statistics (numeric only)
    if numeric_cols:
        eda_report["statistics"] = df[numeric_cols].describe().to_dict()
    else:
        eda_report["statistics"] = {}

    # Target imbalance check (if last column categorical)
    if categorical_cols:
        target_col = categorical_cols[-1]
        eda_report["target_analysis"] = {
            "target_column": target_col,
            "class_distribution": df[target_col].value_counts().to_dict()
        }
    else:
        eda_report["target_analysis"] = {}

    return eda_report
