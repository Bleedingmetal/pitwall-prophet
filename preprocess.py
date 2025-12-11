import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

class F1RacePositionPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = pd.read_csv(self.data_path)
        self.processed_races = None
        self.feature_names = None
        self.team_encoder = None
        
    def wrangle_data(self):
        """Main preprocessing pipeline"""
        print("="*80)
        print("STEP 1: Loading and Initial Exploration")
        print("="*80)
        print(f"Total rows loaded: {self.raw_data.shape[0]}")
        print(f"Total columns: {self.raw_data.shape[1]}")
        print("\n" + str(self.raw_data.head(2)))
        print(".\n.\n.")
        print(str(self.raw_data.tail(2)) + "\n")
        
        # Include DNF drivers, set their position to 21
        print("="*80)
        print("STEP 2: Including DNF Drivers (Position 21)")
        print("="*80)
        initial_count = len(self.raw_data)
        self.raw_data['final_position'] = self.raw_data['final_position'].fillna(21)
        print(f"Total rows: {len(self.raw_data)}")
        print(f"DNF drivers set to position 21: {self.raw_data['final_position'].eq(21).sum()}")
        print(f"Finished drivers: {self.raw_data['final_position'].le(20).sum()}\n")
        
        # Analyze race completeness
        print("="*80)
        print("STEP 3: Analyzing Race Completeness")
        print("="*80)
        race_finishers = self.raw_data.groupby(['year', 'round']).size().reset_index(name='num_finishers')
        print(f"Total unique races: {len(race_finishers)}")
        print(f"\nFinishers distribution:")
        print(race_finishers['num_finishers'].value_counts().sort_index())
        
        # Filter races with exactly 20 finishers (including DNFs as position 21)
        valid_races = race_finishers[
            race_finishers['num_finishers'] == 20
        ][['year', 'round']]
        
        print(f"\nRaces with exactly 20 finishers: {len(valid_races)}")
        
        self.raw_data = self.raw_data.merge(valid_races, on=['year', 'round'], how='inner')
        print(f"Rows after filtering valid races: {len(self.raw_data)}\n")
        
        # Engineer features
        print("="*80)
        print("STEP 4: Feature Engineering")
        print("="*80)
        self._engineer_features()
        
        # Encode teams
        print("\n" + "="*80)
        print("STEP 5: Encoding Team Names")
        print("="*80)
        self._encode_teams()
        
        # Create race-level dataset
        print("\n" + "="*80)
        print("STEP 6: Creating Race-Level Dataset")
        print("="*80)
        self._create_race_tensors()
        
    def _engineer_features(self):
        """Create derived features for prediction"""
        
        # Normalize grid position (1-20 → 0-1), fill missing with median
        self.raw_data['grid_pos_norm'] = (self.raw_data['grid_position'] / 20.0).fillna(0.5)
        print("✓ Created normalized grid position")
        
        # Pit stop features
        self.raw_data['pit_stops_int'] = self.raw_data['pit_stops'].fillna(0).astype(int)
        self.raw_data['total_pit_time_sec'] = self.raw_data['total_pit_time'].fillna(0)
        
        # Compute pit timing as fraction of race
        self.raw_data['first_pit_frac'] = (
            self.raw_data['first_pit_lap'] / self.raw_data['total_laps']
        ).fillna(0)
        self.raw_data['last_pit_frac'] = (
            self.raw_data['last_pit_lap'] / self.raw_data['total_laps']
        ).fillna(0)
        
        # Binary features
        self.raw_data['pitted_at_all'] = (self.raw_data['pit_stops'] > 0).astype(int)
        self.raw_data['pit_before_half'] = (
            self.raw_data['first_pit_frac'] < 0.5
        ).astype(int)
        
        print("✓ Created pit stop features (count, timing, binary flags)")
        
        # Fastest lap features
        self.raw_data['fastest_lap_rank_norm'] = (
            self.raw_data['fastest_lap_rank'] / 20.0
        ).fillna(1.0)  # Missing = worst rank
        
        self.raw_data['fastest_lap_lap_frac'] = (
            self.raw_data['fastest_lap_lap'] / self.raw_data['total_laps']
        ).fillna(0.5)  # Missing = mid-race
        
        # Parse fastest lap time to seconds (handle MM:SS.mmm format)
        def parse_lap_time(time_str):
            if pd.isna(time_str):
                return np.nan
            try:
                parts = str(time_str).split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    return minutes * 60 + seconds
                else:
                    return float(time_str)
            except:
                return np.nan
        
        self.raw_data['fastest_lap_seconds'] = self.raw_data['fastest_lap_time'].apply(parse_lap_time)
        
        # Normalize fastest lap time within each race
        race_groups = self.raw_data.groupby(['year', 'round'])
        self.raw_data['fastest_lap_delta'] = race_groups['fastest_lap_seconds'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        ).fillna(0.5)
        
        print("✓ Created fastest lap features (rank, timing, delta)")
        
        # Summary
        print(f"\nTotal features engineered: {len([c for c in self.raw_data.columns if c.endswith('_norm') or c.endswith('_frac') or c.endswith('_int') or c.endswith('_delta')])}")
        
    def _encode_teams(self):
        """One-hot encode team names"""
        unique_teams = self.raw_data['team_name'].unique()
        print(f"Unique teams found: {len(unique_teams)}")
        print(f"Teams: {sorted(unique_teams)[:10]}...")
        
        # Create one-hot encoding
        team_dummies = pd.get_dummies(self.raw_data['team_name'], prefix='team')
        self.raw_data = pd.concat([self.raw_data, team_dummies], axis=1)
        
        self.team_columns = team_dummies.columns.tolist()
        print(f"✓ Created {len(self.team_columns)} team one-hot features")
        
    def _create_race_tensors(self):
        """Transform data into [num_races, 20, num_features] format"""
        
        # Define feature columns (excluding driver identity and target)
        exclude_cols = [
            'year', 'round', 'race_slug', 'race_name', 'total_laps',
            'driver_number', 'driver_code', 'driver_name', 'team_name',
            'grid_time', 'final_position_raw', 'final_position',
            'laps_completed', 'time_or_status', 'race_time', 'points',
            'pit_stops', 'first_pit_lap', 'last_pit_lap', 'total_pit_time',
            'fastest_lap_rank', 'fastest_lap_lap', 'fastest_lap_time',
            'fastest_lap_avg_speed', 'is_winner', 'fastest_lap_seconds',
            'grid_position'
            # Do NOT exclude dnf_rate_driver or dnf_rate_team
        ]
        
        feature_cols = [
            c for c in self.raw_data.columns 
            if c not in exclude_cols
        ]
        
        self.feature_names = feature_cols
        print(f"Feature columns selected: {len(feature_cols)}")
        print("Features:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"  {i}. {feat}")
        
        # Group by race
        race_groups = self.raw_data.groupby(['year', 'round'])
        
        race_data = []
        race_labels = []
        race_metadata = []
        
        for (year, round_num), race_df in race_groups:
            num_finishers = len(race_df)
            
            # Sort by final position
            race_df = race_df.sort_values('final_position').reset_index(drop=True)
            
            # All races have exactly 20 finishers, no padding needed
            features = race_df[feature_cols].values  # [20, num_features]
            positions = race_df['final_position'].values  # [20,]
            
            race_data.append(features)
            race_labels.append(positions)
            race_metadata.append({
                'year': year,
                'round': round_num,
                'race_name': race_df['race_name'].iloc[0],
                'num_finishers': num_finishers
            })
        
        # Convert to numpy arrays with explicit dtype
        self.X = np.array(race_data, dtype=np.float64)  # [num_races, 20, num_features]
        self.y = np.array(race_labels, dtype=np.float64)  # [num_races, 20]
        self.race_metadata = race_metadata
        
        print(f"\n✓ Created race tensors:")
        print(f"  X shape: {self.X.shape} (num_races, num_drivers, num_features)")
        print(f"  y shape: {self.y.shape} (num_races, num_drivers)")
        print(f"  Total races: {len(race_metadata)}")
        
        # Summary statistics
        print(f"\nRace metadata sample:")
        for i in range(min(3, len(race_metadata))):
            meta = race_metadata[i]
            print(f"  Race {i+1}: {meta['year']} Round {meta['round']} - {meta['race_name'][:50]}... ({meta['num_finishers']} finishers)")
    
    def split_data(self, test_size=0.2, random_state=339):
        """Split data into train/test sets (no validation needed - using CV)"""
        print("\n" + "="*80)
        print("STEP 7: Splitting Data")
        print("="*80)
        
        n_races = len(self.X)
        indices = np.arange(n_races)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # Calculate split point (80/20 train/test)
        test_split = int(n_races * (1 - test_size))
        
        train_idx = indices[:test_split]
        test_idx = indices[test_split:]
        
        self.X_train = self.X[train_idx]
        self.y_train = self.y[train_idx]
        
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        
        print(f"Training set size: {len(self.X_train)} races")
        print(f"Test set size: {len(self.X_test)} races")
        
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}\n")
        
    def save_processed_data(self, output_path='f1_processed_2021_2025.npz'):
        """Save processed data to file"""
        print("="*80)
        print("STEP 8: Saving Processed Data")
        print("="*80)
        
        np.savez_compressed(
            output_path,
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            feature_names=self.feature_names,
            race_metadata=self.race_metadata
        )
        
        print(f"✓ Saved processed data to {output_path}")
        print(f"  File size: {np.round(np.array([self.X_train.nbytes, self.y_train.nbytes]).sum() / 1024**2, 2)} MB")

def main():
    preprocessor = F1RacePositionPreprocessor('f1_processed_2021_2025.csv')
    preprocessor.wrangle_data()
    preprocessor.split_data()
    preprocessor.save_processed_data()
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Load processed data with: np.load('f1_processed_2021_2025.npz')")
    print("2. Build neural network for position prediction")
    print("3. Implement permutation-invariant architecture")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
