"""
NOx Conversion Efficiency Predictor
Reusable prediction helper for the trained model.
"""

import json, joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"

# Load model artifacts
try:
    _model = joblib.load(MODEL_DIR / "nox_model.pkl")
    _scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    with open(MODEL_DIR / "feature_list.json", "r") as f:
        _features = json.load(f)
    print(f"Model loaded successfully with {len(_features)} features")
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model artifacts not found. Run 'python train_model.py' first. Missing: {e}")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

# Dynamically create feature descriptions and ranges based on the loaded features
FEATURE_DESCRIPTIONS = {f: f.replace('_', ' ').title() for f in _features}

# Calculate feature ranges from the sample data for initial validation
try:
    sample_df_for_ranges = pd.read_excel('data/sample_data.xlsx').drop(columns=['NOX_CE'], errors='ignore')
    FEATURE_RANGES = {}
    for col in sample_df_for_ranges.columns:
        if pd.api.types.is_numeric_dtype(sample_df_for_ranges[col]):
            min_val = sample_df_for_ranges[col].min()
            max_val = sample_df_for_ranges[col].max()
            # Add a small buffer to min/max
            buffer = (max_val - min_val) * 0.1 
            if buffer == 0: # Handle cases where min_val == max_val
                buffer = 1 # Provide a default buffer
            FEATURE_RANGES[col] = (min_val - buffer, max_val + buffer)
        else:
            # For non-numeric types, set a generic range or skip if not used in model
            FEATURE_RANGES[col] = (None, None) # Indicate no numerical range
except FileNotFoundError:
    print("Warning: data/sample_data.xlsx not found. Using default ranges for now.")
    FEATURE_RANGES = {f: (0, 100) for f in _features} # Fallback generic ranges

# Ensure all features in _features have a range, even if None,None for non-numeric
for f in _features:
    if f not in FEATURE_RANGES:
        FEATURE_RANGES[f] = (None, None)

def validate_inputs(user_dict: dict) -> tuple[bool, str]:
    """Validate user inputs are within reasonable ranges"""
    for feature, value in user_dict.items():
        if feature in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[feature]
            try:
                val_float = float(value)
                if not (min_val <= val_float <= max_val):
                    return False, f"{feature}: {val_float} outside range [{min_val}, {max_val}]"
            except (ValueError, TypeError):
                return False, f"{feature}: '{value}' is not a valid number"
    return True, "Valid"

def _prepare_input(user_dict: dict) -> np.ndarray:
    """Prepare input for model prediction"""
    # Check for missing features
    missing_features = [f for f in _features if f not in user_dict]
    if missing_features:
        raise KeyError(f"Missing required features: {missing_features}")

    # Convert to float and create array
    try:
        row = [float(user_dict[f]) for f in _features]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Non-numeric input detected: {e}")

    return np.array(row).reshape(1, -1)

def predict_nox_efficiency(user_dict: dict) -> dict:
    """
    Predict NOx conversion efficiency from input parameters

    Args:
        user_dict: Dictionary with feature values

    Returns:
        Dictionary with prediction and metadata
    """
    # Validate inputs
    is_valid, msg = validate_inputs(user_dict)
    if not is_valid:
        raise ValueError(f"Input validation failed: {msg}")

    # Prepare input
    X = _prepare_input(user_dict)

    # Scale features
    X_scaled = _scaler.transform(X)

    # Make prediction
    prediction = float(_model.predict(X_scaled)[0])

    # Clip to physical bounds
    prediction = max(0.0, min(100.0, prediction))

    # Get prediction confidence (using tree variance)
    if hasattr(_model, 'estimators_'):
        # For RandomForest, calculate prediction std across trees
        tree_predictions = [tree.predict(X_scaled)[0] for tree in _model.estimators_]
        prediction_std = np.std(tree_predictions)
        confidence = max(0, 100 - prediction_std * 10)  # Heuristic confidence
    else:
        confidence = 85  # Default confidence

    # Determine efficiency category
    if prediction >= 95:
        category = "Excellent"
    elif prediction >= 90:
        category = "Good"
    elif prediction >= 80:
        category = "Fair"
    else:
        category = "Poor"

    return {
        'nox_ce': round(prediction, 2),
        'confidence': round(confidence, 1),
        'category': category,
        'prediction_std': round(prediction_std if 'prediction_std' in locals() else 0, 2)
    }

def get_feature_info() -> dict:
    """Get information about required features"""
    return {
        'features': _features,
        'descriptions': FEATURE_DESCRIPTIONS,
        'ranges': FEATURE_RANGES
    }

# CLI test function
if __name__ == "__main__":
    print("Testing predictor with sample data...")

    try:
        sample_df = pd.read_excel('data/sample_data.xlsx')
        sample_inputs_df = sample_df.drop(columns=['NOX_CE'], errors='ignore')
        
        # Get the first row as a dictionary
        if not sample_inputs_df.empty:
            sample_inputs = sample_inputs_df.iloc[0].to_dict()
            try:
                result = predict_nox_efficiency(sample_inputs)
                print(f"Prediction successful!")
                print(f"NOx CE: {result['nox_ce']}%")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']}%")
            except Exception as e:
                print(f"Prediction failed: {e}")
        else:
            print("Warning: Sample data is empty, cannot create sample_inputs for testing.")
            print("Skipping prediction test due to empty sample inputs.")
    except FileNotFoundError:
        print("Error: data/sample_data.xlsx not found. Cannot run CLI test.")
    except Exception as e:
        print(f"An unexpected error occurred during CLI test: {e}")
