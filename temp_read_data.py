import pandas as pd
import pathlib

DATA_PATH = pathlib.Path("data") / "nox_data.xlsx"

try:
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    print("First 200 rows of the dataset:")
    print(df.head(200).to_string())
except Exception as e:
    print(f"Error reading data: {e}")
