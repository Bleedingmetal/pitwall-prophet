import pandas as pd

# Load raw data
df = pd.read_csv('f1_raw_2021_2025.csv')

# Mark DNF: final_position is NaN or time_or_status is 'DNF' or 'NC'
df['is_dnf'] = df['final_position'].isna() | df['time_or_status'].astype(str).str.upper().isin(['DNF', 'NC'])

driver_dnf = df.groupby('driver_name')['is_dnf'].mean().reset_index().rename(columns={'is_dnf': 'dnf_rate_driver'})
team_dnf = df.groupby('team_name')['is_dnf'].mean().reset_index().rename(columns={'is_dnf': 'dnf_rate_team'})

# Merge rates back to each row
df = df.merge(driver_dnf, on='driver_name', how='left')
df = df.merge(team_dnf, on='team_name', how='left')

# Save annotated CSV
df.to_csv('f1_raw_2021_2025_with_dnf.csv', index=False)
print('Annotated DNF rates and saved to f1_raw_2021_2025_with_dnf.csv')

# Load processed CSV
csv_path = "f1_processed_2021_2025.csv"
df = pd.read_csv(csv_path)

# Identify possible DNF columns (boolean columns not team or is_winner)
exclude_cols = [
    "year", "round", "race_slug", "race_name", "driver_code", "driver_name", "team_name",
    "grid_pos", "grid_pos_norm", "pit_stops_int", "first_pit_frac", "last_pit_frac",
    "pitted_at_all", "pit_before_half", "total_pit_time_sec", "fastest_lap_rank_norm",
    "fastest_lap_lap_frac", "fastest_lap_delta", "is_winner"
]

# Find boolean columns
bool_cols = [c for c in df.columns if c not in exclude_cols and df[c].dropna().isin([True, False]).all()]

# Heuristic: DNF column has relatively few True values, and for drivers with high final_position
if bool_cols:
    dnf_col = bool_cols[0]
else:
    raise RuntimeError("No DNF column found!")

# Compute DNF rate per driver
dnf_rates_driver = df.groupby("driver_name")[dnf_col].mean()
# Compute DNF rate per team
dnf_rates_team = df.groupby("team_name")[dnf_col].mean()

# Annotate each row with the driver's and team's DNF rate
df["dnf_rate_driver"] = df["driver_name"].map(dnf_rates_driver)
df["dnf_rate_team"] = df["team_name"].map(dnf_rates_team)

# Save annotated CSV
out_path = "f1_processed_2021_2025_with_dnf_driver_team.csv"
df.to_csv(out_path, index=False)
print(f"Annotated CSV written to {out_path}")
