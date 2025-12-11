import pandas as pd

# Load annotated raw data with DNF rates
df = pd.read_csv('f1_raw_2021_2025_with_dnf.csv')

# Save as processed CSV for pipeline
df.to_csv('f1_processed_2021_2025.csv', index=False)
print('Created f1_processed_2021_2025.csv with DNF rates included.')
