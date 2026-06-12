

import os
import json
import pandas as pd


def process_excel_with_mappings(
    xlsx_path: str,
    columns: list[str],
    output_dir: str
) -> pd.DataFrame:
    """
    Reads an Excel file, creates/updates per-column JSON mapping files,
    and replaces values in the DataFrame using those mappings.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel file.
    columns : list[str]
        Columns to process.
    output_dir : str
        Directory where JSON files are stored/created.

    Returns
    -------
    pd.DataFrame
        DataFrame with mapped values.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read Excel
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    for col in columns:
        if col not in df.columns:
            continue  # silently skip missing columns

        json_path = os.path.join(output_dir, f"{col}.json")

        # Load existing mapping if present
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            mapping = {}

        # Extract unique string values in column (excluding NaN)
        values = df[col].dropna().unique()
        string_values = [v for v in values if isinstance(v, str)]

        # Add missing values to mapping (identity mapping)
        updated = False
        for v in string_values:
            if v not in mapping:
                mapping[v] = v
                updated = True

        # Save mapping only if new entries were added or file didn't exist
        if updated or not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False, indent=2)

        # Replace values in DataFrame using mapping (only for strings)
        df[col] = df[col].apply(
            lambda x: mapping.get(x, x) if isinstance(x, str) else x
        )

    return df

file_path = "../Data extraction Dutch cNLP tools 10Jun2026.xlsx"
process_excel_with_mappings(file_path,['Dev region','Ev region','NLP Task description','Dev size','Ev size'], 'raw_data_mappings')
