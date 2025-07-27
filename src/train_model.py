import os
import sys
from pathlib import Path

# Add the project root to sys.path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

def train_and_save_model(data_path='data/sample_data.xlsx'):
    """
    Trains a machine learning model to predict NOX conversion efficiency,
    evaluates its performance, and saves the trained model, scaler, and feature list.

    Args:
        data_path (str, optional): Path to the dataset (Excel or CSV).
                                   Defaults to 'data/sample_data.xlsx'.
    """
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

    # Define features (X) and target (y)
    # The target column is 'NOX_CE'
    target_column = 'NOX_CE'
    
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle potential duplicate columns after concatenation if any were missed
    X = X.loc[:,~X.columns.duplicated()]

    # Drop any columns that are all NaN (might happen with sample data if mapping is off)
    X = X.dropna(axis=1, how='all')

    # Impute missing values for simplicity
    for col in X.columns:
        if X[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                # For non-numeric missing values, fill with mode or a placeholder
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'UNKNOWN')

    # Data Analysis: Feature Correlation with Target
    print("\n--- Feature Correlation with NOX_CE ---")
    # Concatenate X and y to calculate correlations
    df_for_corr = pd.concat([X, y], axis=1)
    correlations = df_for_corr.corr()[target_column].sort_values(ascending=False)
    print(correlations)
    print("------------------------------------")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a RandomForestRegressor model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the trained model, scaler, and feature list
    model_path = 'models/nox_model.pkl'
    scaler_path = 'models/feature_scaler.pkl'
    feature_list_path = 'models/feature_list.json'

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(feature_list_path, 'w') as f:
        json.dump(X.columns.tolist(), f)

    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    importance_path = 'models/feature_importance.csv'
    feature_importances.to_csv(importance_path, index=False)
    print(f"Feature importances saved to {importance_path}")

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Feature list saved to {feature_list_path}")
    print("Model training and saving complete.")

if __name__ == '__main__':
    # You can call train_and_save_model() with your actual data file later
    # For now, it will use the sample data.
    train_and_save_model()
    # To run with real data, uncomment and modify the line below:
    # train_and_save_model(data_path='path/to/your/actual_data.csv')
