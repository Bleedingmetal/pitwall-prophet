import pandas as pd

# Load both CSVs
raw = pd.read_csv('f1_raw_2021_2025.csv')
proc = pd.read_csv('f1_processed_2021_2025.csv')

# We'll match on year, round, driver_name
merge_cols = ['year', 'round', 'driver_name']

# Only keep the final_position column from raw
to_merge = raw[merge_cols + ['final_position']]

# Merge into processed
proc_merged = pd.merge(proc, to_merge, on=merge_cols, how='left')

# Save back to processed CSV (overwrite or backup as needed)
proc_merged.to_csv('f1_processed_2021_2025.csv', index=False)
print('Added final_position column to f1_processed_2021_2025.csv')
