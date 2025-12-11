# Pitwall Prophet UI

A web-based interface for the F1 Race Position Prediction System.

## Architecture

- **Backend API** (`api.py`): Flask REST API that inherits from `F1PositionPredictionPipeline`
- **Frontend UI** (`ui.html`): Single-page web application with real-time predictions
- **Main Pipeline** (`main.py`): Unchanged, provides base functionality

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements_ui.txt
```

### 2. Start the Backend API

```bash
python api.py
```

The API will:
- Initialize the prediction pipeline
- Train the model with best hyperparameters
- Start serving on `http://localhost:5000`

### 3. Open the UI

Open `ui.html` in your web browser:
```bash
open ui.html  # macOS
# or
xdg-open ui.html  # Linux
# or just double-click the file on Windows
```

## Features

### Backend API Endpoints

- `GET /api/health` - Check API status
- `GET /api/features` - Get feature template
- `GET /api/sample_data` - Load random test race
- `POST /api/predict` - Predict race positions

### UI Features

1. **Load Sample Race**: Loads a random race from the test set with actual driver data
2. **Manual Input**: Edit driver names, teams, and key features
3. **Predict Positions**: Get ML predictions for final race positions
4. **Visual Results**: Beautiful gradient UI with podium highlighting

## Usage

1. Click "Load Sample Race" to populate with real F1 data
2. Modify features as needed (grid position, pit stops, etc.)
3. Click "Predict Positions" to see predicted finishing order
4. Results show predicted positions with podium highlighting (Gold/Silver/Bronze)

## API Request Format

```json
{
  "drivers": [
    {
      "driver_name": "Max Verstappen",
      "team_name": "Red Bull Racing",
      "features": {
        "grid_position": 1,
        "grid_pos_norm": 0.05,
        "points": 575,
        ...
      }
    },
    ... (20 drivers total)
  ]
}
```

## API Response Format

```json
{
  "predictions": [
    {
      "driver_name": "Max Verstappen",
      "team_name": "Red Bull Racing",
      "predicted_position": 1
    },
    ...
  ],
  "status": "success"
}
```

## Technical Details

- **Model**: Uses the same PositionPredictionNetwork from `main.py`
- **Hyperparameters**: (64, 32) hidden layers, relu activation, 50 epochs
- **Accuracy**: ~64% within ±2 positions, ~22% exact position
- **Features**: 53 features per driver including grid position, team encoding, pit stops, etc.

## Notes

- The API trains the model on startup (takes ~2 minutes)
- All 20 drivers must be provided for prediction
- Features are normalized according to the preprocessing pipeline
- Hungarian algorithm ensures unique position assignments
