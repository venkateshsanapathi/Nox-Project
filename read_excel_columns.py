import pandas as pd

try:
    # Attempt to read the Excel file, assuming header is in the first row (index 0)
    df = pd.read_excel('data/nox_data.xlsx', header=0)
    print("Columns from data/nox_data.xlsx:")
    for col in df.columns:
        print(col)
except Exception as e:
    print(f"Error reading Excel file: {e}")
