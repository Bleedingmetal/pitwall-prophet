from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import torch
from main import F1PositionPredictionPipeline, PositionPredictionNetwork
import json

app = Flask(__name__)
CORS(app)

api_pipeline = None



class F1PredictionAPI(F1PositionPredictionPipeline):
    """Extended pipeline with API inference capabilities"""
    
    def __init__(self, data_file, model_path=None, device='cpu', seed=339):
        super().__init__(data_file, device, seed)
        self.model_path = model_path
        
    def load_trained_model(self, params):
        """Load a pre-trained model"""
        input_dim = self.X_train.shape[2]
        self.model = PositionPredictionNetwork(
            input_dim=input_dim,
            hidden_layers=params['hidden_layers'],
            dropout=params['dropout'],
            activation=params['activation']
        ).to(self.device)
        
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.model.eval()
        return self.model
    
    def predict_race_positions(self, race_features):
        """
        Predict positions for a single race
        
        Args:
            race_features: [20, num_features] numpy array
        
        Returns:
            predictions: dict with driver positions
        """
        if not hasattr(self, 'model'):
            raise ValueError("Model not loaded. Call load_trained_model() first.")
        
        # Convert to tensor
        race_tensor = torch.from_numpy(race_features).float().unsqueeze(0)  # [1, 20, features]
        race_tensor = race_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(race_tensor)
            probs = torch.softmax(logits, dim=-1)
            assignments = self.hungarian_matching(probs)
        
        return assignments[0]  # Return first batch
    
    def get_feature_template(self):
        return {
            'feature_names': list(self.feature_names),
            'num_drivers': 20,
            'num_features': len(self.feature_names),
            'example_structure': {
                'driver_0': {name: 0.0 for name in self.feature_names},
                'driver_1': {name: 0.0 for name in self.feature_names},
                # ... up to driver_19
            }
        }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api_pipeline is not None and hasattr(api_pipeline, 'model')
    })


@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature template"""
    if api_pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    return jsonify(api_pipeline.get_feature_template())


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict race positions
    
    Expected input format:
    {
        "drivers": [
            {
                "driver_name": "Max Verstappen",
                "team_name": "Red Bull Racing",
                "features": {
                    "grid_position": 1,
                    "grid_pos_norm": 0.05,
                    ...
                }
            },
            ... (20 drivers total)
        ]
    }
    """
    if api_pipeline is None or not hasattr(api_pipeline, 'model'):
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        drivers = data.get('drivers', [])
        
        if len(drivers) != 20:
            return jsonify({'error': f'Expected 20 drivers, got {len(drivers)}'}), 400
        
        # Convert driver features to numpy array
        race_features = np.zeros((20, len(api_pipeline.feature_names)))
        driver_info = []
        
        for i, driver in enumerate(drivers):
            driver_info.append({
                'name': driver.get('driver_name', f'Driver {i+1}'),
                'team': driver.get('team_name', 'Unknown')
            })
            
            features = driver.get('features', {})
            for j, feature_name in enumerate(api_pipeline.feature_names):
                race_features[i, j] = features.get(feature_name, 0.0)
        
        # Get predictions
        predictions = api_pipeline.predict_race_positions(race_features)
        
        # Format response
        results = []
        for i in range(20):
            results.append({
                'driver_name': driver_info[i]['name'],
                'team_name': driver_info[i]['team'],
                'predicted_position': int(predictions[i]),
                'actual_position': drivers[i].get('actual_position', 0)  # Include actual position if available
            })
        
        # Sort by predicted position
        results.sort(key=lambda x: x['predicted_position'])
        
        return jsonify({
            'predictions': results,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample_data', methods=['GET'])
def get_sample_data():
    """Get sample race data from test set (2024-2025 only)"""
    if api_pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        # Filter for races from 2024 and 2025 only
        recent_race_indices = []
        for idx, meta in enumerate(api_pipeline.race_metadata):
            if idx >= len(api_pipeline.race_metadata) - len(api_pipeline.X_test):
                test_idx = idx - (len(api_pipeline.race_metadata) - len(api_pipeline.X_test))
                if int(meta['year']) >= 2024:
                    recent_race_indices.append(test_idx)
        
        if not recent_race_indices:
            # Fallback to any test race if no 2024/2025 races found
            race_idx = np.random.randint(0, len(api_pipeline.X_test))
        else:
            # Get a random race from 2024 or 2025
            race_idx = np.random.choice(recent_race_indices)
        
        race_features = api_pipeline.X_test[race_idx]
        race_targets = api_pipeline.y_test[race_idx]
        
        # Load raw data for driver names and grid positions
        raw_df = pd.read_csv('f1_raw_2021_2025.csv')
        race_meta = api_pipeline.race_metadata[len(api_pipeline.race_metadata) - len(api_pipeline.X_test) + race_idx]
        
        # Get ALL drivers from this race (including DNFs) for starting grid
        race_df_all = raw_df[
            (raw_df['year'] == int(race_meta['year'])) & 
            (raw_df['round'] == int(race_meta['round']))
        ].copy()
        
        # Get finishers sorted by final position (matches model data order)
        race_df_finishers = race_df_all[race_df_all['final_position'].notna()].sort_values('final_position').reset_index(drop=True)
        
        # Create feature mapping for finishers (indexed by their finish position)
        finisher_features = {}
        for idx, row in race_df_finishers.iterrows():
            if pd.notna(row['grid_position']):
                grid_pos = int(float(row['grid_position']))
                features = {}
                for j, feature_name in enumerate(api_pipeline.feature_names):
                    features[feature_name] = float(race_features[idx, j])
                finisher_features[grid_pos] = {
                    'features': features,
                    'actual_position': int(race_targets[idx]) if race_targets[idx] <= 20 else 0
                }
        
        # Build complete drivers list from all race participants sorted by grid position
        race_df_by_grid = race_df_all[race_df_all['grid_position'].notna()].sort_values('grid_position').reset_index(drop=True)
        
        drivers = []
        for _, row in race_df_by_grid.iterrows():
            grid_pos = int(float(row['grid_position']))
            
            # Get features from finisher data if available, otherwise use zeros
            if grid_pos in finisher_features:
                features = finisher_features[grid_pos]['features']
                actual_pos = finisher_features[grid_pos]['actual_position']
            else:
                # DNF driver - no features available from model
                features = {name: 0.0 for name in api_pipeline.feature_names}
                actual_pos = 0
            
            drivers.append({
                'driver_name': row['driver_name'],
                'team_name': row['team_name'],
                'grid_position': grid_pos,
                'actual_position': actual_pos,
                'features': features
            })
        
        # Only pad if we have fewer than 20 drivers
        while len(drivers) < 20:
            grid_pos = len(drivers) + 1
            features = {name: 0.0 for name in api_pipeline.feature_names}
            drivers.append({
                'driver_name': f'Grid Position {grid_pos}',
                'team_name': 'Unknown',
                'grid_position': grid_pos,
                'actual_position': 0,
                'features': features
            })
        
        # Clean up race name to remove "RACE RESULT" suffix
        race_name = race_meta['race_name']
        race_name = race_name.replace(' - RACE RESULT', '').replace(' RACE RESULT', '')
        
        return jsonify({
            'race_info': {
                'year': int(race_meta['year']),
                'round': int(race_meta['round']),
                'name': race_name
            },
            'drivers': drivers
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """
    Upload CSV file with race data
    Expected format: CSV with driver_name, team_name, and feature columns
    """
    if api_pipeline is None or not hasattr(api_pipeline, 'model'):
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get CSV data from request
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
        elif 'csv_data' in request.json:
            from io import StringIO
            csv_data = request.json['csv_data']
            df = pd.read_csv(StringIO(csv_data))
        else:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        # Validate we have enough drivers
        if len(df) < 15:
            return jsonify({'error': f'Need at least 15 drivers, got {len(df)}'}), 400
        
        # Pad to 20 drivers if needed
        while len(df) < 20:
            df = pd.concat([df, pd.DataFrame([{col: 0 for col in df.columns}])], ignore_index=True)
        
        # Take first 20 drivers
        df = df.head(20)
        
        # Extract driver info and features
        drivers = []
        race_features = np.zeros((20, len(api_pipeline.feature_names)))
        
        for i in range(20):
            driver_name = df.iloc[i].get('driver_name', f'Driver {i+1}')
            team_name = df.iloc[i].get('team_name', 'Unknown')
            
            drivers.append({
                'driver_name': str(driver_name),
                'team_name': str(team_name)
            })
            
            # Map CSV columns to features
            for j, feature_name in enumerate(api_pipeline.feature_names):
                if feature_name in df.columns:
                    race_features[i, j] = float(df.iloc[i][feature_name]) if pd.notna(df.iloc[i][feature_name]) else 0.0
                else:
                    race_features[i, j] = 0.0
        
        # Get predictions
        predictions = api_pipeline.predict_race_positions(race_features)
        
        # Format response
        results = []
        for i in range(20):
            results.append({
                'driver_name': drivers[i]['driver_name'],
                'team_name': drivers[i]['team_name'],
                'predicted_position': int(predictions[i])
            })
        
        # Sort by predicted position
        results.sort(key=lambda x: x['predicted_position'])
        
        return jsonify({
            'predictions': results,
            'status': 'success',
            'message': f'Processed {len(df)} drivers from CSV'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_template', methods=['GET'])
def download_template():
    """Download CSV template with all required features"""
    if api_pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 500
    
    try:
        # Create template DataFrame
        template_data = {
            'driver_name': ['Max Verstappen', 'Lewis Hamilton', 'Charles Leclerc'],
            'team_name': ['Red Bull Racing', 'Ferrari', 'Ferrari']
        }
        
        # Add all feature columns with example values
        for feature in api_pipeline.feature_names:
            if 'grid_position' in feature:
                template_data[feature] = [1, 2, 3]
            elif 'points' in feature:
                template_data[feature] = [575, 450, 400]
            elif 'norm' in feature:
                template_data[feature] = [0.05, 0.1, 0.15]
            else:
                template_data[feature] = [0.0, 0.0, 0.0]
        
        df = pd.DataFrame(template_data)
        csv_string = df.to_csv(index=False)
        
        return jsonify({
            'csv': csv_string,
            'feature_count': len(api_pipeline.feature_names),
            'features': list(api_pipeline.feature_names)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def initialize_api():
    """Initialize the API with trained model"""
    global api_pipeline
    
    print("Initializing F1 Position Prediction API...")
    
    # Initialize pipeline
    api_pipeline = F1PredictionAPI(
        data_file='f1_processed_2021_2025.npz',
        device='cuda',
        seed=339
    )
    
    # Load data
    api_pipeline.load_data()
    
    # Define best parameters
    best_params = {
        'hidden_layers': (64, 32),
        'dropout': 0.2,
        'activation': 'relu',
        'lr': 1e-3,
        'optimizer': 'adam',
        'batch_size': 32,
        'weight_decay': 1e-4,
        'epochs': 50
    }
    
    # Train or load model
    print("Training model...")
    api_pipeline.train_final_model(best_params)
    
    print("API initialized successfully!")


if __name__ == '__main__':
    initialize_api()
    app.run(host='0.0.0.0', port=5001, debug=False)
