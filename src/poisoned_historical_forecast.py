# poisoned_historical_forecast.py

import torch
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pathlib
from tqdm import tqdm # Import tqdm for progress bar
from sklearn.preprocessing import MinMaxScaler # Import MinMaxScaler for type checking
import argparse # Added for command-line arguments
import sys

# --- Determine experiment directory and Import Model Utils --- (NEW)
script_dir = pathlib.Path(__file__).parent.resolve()
print(f"Script directory: {script_dir}")

# --- Import Model Utils --- 
from model_utils import (
        select_meteo_features,
        select_temporal_features,
        create_time_features,
        # create_sequences, # Not strictly needed for prediction loop
        # TimeSeriesDataset, # Not strictly needed for prediction loop  
        LSTMForecastModel2,       
    )
print(f"Successfully imported model_utils from script directory.")

# --- Argument Parsing --- (NEW)
parser = argparse.ArgumentParser(description="Hyperparameter tuning for time series models.")
parser.add_argument('--target', type=str, required=True, help='Target series name (e.g., consumption, pv_production)')
parser.add_argument('--model', type=str, required=True, choices=['LSTM', 'GRU', 'Transformer'], help='Model type (LSTM, GRU, Transformer)')
parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon in hours')
parser.add_argument('--sequence_length', type=int, required=True, help='Sequence length')
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value for FGSM')
parser.add_argument('--poisoning_ratio', type=float, required=True, help='Poisoning ratio')
args = parser.parse_args()
print(f"Arguments: {args}")

# --- Configuration from Arguments --- (NEW)
TARGET_COLUMN = args.target
MODEL_TYPE = args.model
FORECAST_HORIZON = args.horizon
SEQUENCE_LENGTH = args.sequence_length
TIMESTAMP_COLUMN = 'time' # Keep fixed
RANDOM_SEED = args.seed # Keep fixed for tuning reproducibility
PROCESS_TYPE = 'poisoning'
EPSILON = args.epsilon
POISONING_RATIO = args.poisoning_ratio
ATTACK_TYPE = 'FGSM'

# Print effective configuration being used
print("\n--- Effective Configuration (from args) ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Process Type: {PROCESS_TYPE}")
print(f"Epsilon: {EPSILON}")
print(f"Poisoning Ratio: {POISONING_RATIO}")

print("--- Starting Historical Forecast Generation ---")

# --- Configuration (Set based on extracted path info) ---

DATA_PATH = '../../data/dataset.csv' # Keep this constant for now

EXPERIMENTS_BASE_PATH = script_dir / 'experiments'
# Specific path for this run's tuning artifacts
CLEAN_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / 'clean_training_artifacts'
print(f"Clean Artifacts load Path: {CLEAN_ARTIFACTS_LOAD_PATH}")

POISONED_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / f'{PROCESS_TYPE}' / f'{ATTACK_TYPE}' / f'epsilon_{EPSILON}' / f'ratio_{POISONING_RATIO}'

# Define paths based on script location and extracted params
MODEL_LOAD_PATH = POISONED_LOAD_PATH / 'models' # Match train_model.py format
ARTIFACTS_PATH = POISONED_LOAD_PATH / 'artifacts' # Load artifacts from here, save predictions here
JSON_PATH = POISONED_LOAD_PATH / 'json_files' # save predictions here
os.makedirs(JSON_PATH, exist_ok=True)

UPPER_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / 'tuning_artifacts' #  load from this path


# Check essential config
if TARGET_COLUMN == 'unknown' or MODEL_TYPE == 'unknown' or FORECAST_HORIZON is None or PROCESS_TYPE == 'unknown':
    print("Error: Essential configuration (target, model_type, horizon, process_type) could not be determined from path.")
    exit()

# Define features dynamically
METEO_FEATURES = select_meteo_features(TARGET_COLUMN)
TEMPORAL_FEATURES = select_temporal_features(TARGET_COLUMN) # Base temporal features
INPUT_FEATURES = METEO_FEATURES + TEMPORAL_FEATURES + [TARGET_COLUMN]

# Load training config to get SEQUENCE_LENGTH
# Load training config to get SEQUENCE_LENGTH and other potential parameters
training_config_path = ARTIFACTS_PATH / "training_config.json"
try:
    with open(training_config_path, 'r') as f:
        training_config = json.load(f)
    SEQUENCE_LENGTH = training_config.get('SEQUENCE_LENGTH', 24) # Default if not found
    print(f"Loaded SEQUENCE_LENGTH ({SEQUENCE_LENGTH}) from training_config.json")
except FileNotFoundError:
    print(f"Warning: training_config.json not found at {training_config_path}. Using default SEQUENCE_LENGTH=24.")
    SEQUENCE_LENGTH = 24
except Exception as e:
    print(f"Error loading training_config.json: {e}. Using default SEQUENCE_LENGTH=24.")
    SEQUENCE_LENGTH = 24


TIMESTAMP_COLUMN = 'time' # Assuming this is always the same
PREDICTION_STRIDE = 1 # Step size for the sliding window
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Print effective configuration being used
print("\n--- Effective Historical Forecast Configuration ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Process Type: {PROCESS_TYPE}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Model Load Path: {MODEL_LOAD_PATH}")
print(f"Artifacts Path: {ARTIFACTS_PATH}")
print(f"Device: {DEVICE}")

# Ensure artifacts directory exists (needed for loading)
if not ARTIFACTS_PATH.exists():
    print(f"Error: Artifacts directory not found at {ARTIFACTS_PATH}. Run tuning and training first.")
    exit()

# --- Load Scalers ---
print("\n--- Loading Scalers ---")
feature_scaler_path = UPPER_ARTIFACTS_LOAD_PATH / "feature_scaler.joblib"
target_scaler_path = UPPER_ARTIFACTS_LOAD_PATH / "target_scaler.joblib"
try:
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print(f"Loaded feature scaler from {feature_scaler_path}")
    print(f"Loaded target scaler from {target_scaler_path}")
    # Optional check for target scaler type
    if not isinstance(target_scaler, MinMaxScaler):
        print(f"Warning: 'target_scaler' is of type {type(target_scaler)}, expected MinMaxScaler.")
except FileNotFoundError:
    print(f"Error: Scaler file not found. Run tune_hyperparameters.py first.")
    exit()

# --- Load Best Model ---
print("\n--- Loading Best Model ---")
if not MODEL_LOAD_PATH.exists():
     print(f"Error: Model save directory not found at {MODEL_LOAD_PATH}. Run train_model.py first.")
     exit()
try:
    checkpoint_files = list(MODEL_LOAD_PATH.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")
    best_model_path = max(checkpoint_files, key=os.path.getctime)
    print(f"Found best model checkpoint: {best_model_path}")
except FileNotFoundError:
    print(f"Error: No model checkpoint (.ckpt) found in {MODEL_LOAD_PATH}. Run train_model.py first.")
    exit()
except Exception as e:
    print(f"Error finding checkpoint file: {e}")
    exit()

# Load the model from the checkpoint dynamically
try:
    if MODEL_TYPE == 'LSTM':
        model = LSTMForecastModel2.load_from_checkpoint(best_model_path)
    elif MODEL_TYPE == 'GRU':
        model = GRUForecastModel2.load_from_checkpoint(best_model_path)
    elif MODEL_TYPE == 'Transformer':
        model = TransformerForecastModel.load_from_checkpoint(best_model_path)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE for loading: {MODEL_TYPE}")

    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print(f"{MODEL_TYPE} model loaded successfully from {best_model_path} and set to evaluation mode.")
except Exception as e:
    print(f"Error loading {MODEL_TYPE} model from checkpoint {best_model_path}: {e}")
    exit()

# --- Load and Preprocess Full Data ---
print("\n--- Loading and Preprocessing Full Data ---")
df = pd.read_csv(DATA_PATH,
                 parse_dates=[TIMESTAMP_COLUMN],
                 index_col=[TIMESTAMP_COLUMN])
print(f"Loaded dataset with shape: {df.shape}")

# Generate time features (pass TARGET_COLUMN)
time_features_df = create_time_features(df.index, TARGET_COLUMN)
df = pd.concat([df, time_features_df], axis=1)
print(f"Added temporal features. Shape: {df.shape}")

# Handle missing values
if df.isnull().sum().any():
    print("Missing values found. Applying forward fill.")
    df = df.ffill()
else:
    print("No missing values found.")

# Select relevant columns (including target for scaling)
df = df[INPUT_FEATURES].copy() # Use copy to avoid SettingWithCopyWarning
print(f"Selected relevant columns. Shape: {df.shape}")

# --- Scale Full Data ---
print("\n--- Scaling Full Data using Loaded Scalers ---")
try:
    scaled_features_full = feature_scaler.transform(df[INPUT_FEATURES])
    scaled_target_full = target_scaler.transform(df[[TARGET_COLUMN]])
except ValueError as e:
    print(f"Error during scaling: {e}")
    print("Ensure columns match those used for fitting scalers.")
    raise e

# Create scaled DataFrame
df_full_scaled = pd.DataFrame(scaled_features_full, columns=INPUT_FEATURES, index=df.index)
# Overwrite the target column with its scaled version
df_full_scaled[TARGET_COLUMN] = scaled_target_full
print(f"Full dataset scaled. Shape: {df_full_scaled.shape}")

# --- Historical Forecasting Loop ---
print(f"\n--- Starting Prediction Loop (Stride={PREDICTION_STRIDE}) ---")

all_predictions_list = [] # List to store prediction dictionaries

# Calculate the last possible start index for a full sequence
last_possible_start_index = len(df_full_scaled) - SEQUENCE_LENGTH
num_prediction_points = (last_possible_start_index // PREDICTION_STRIDE) + 1

print(f"Predicting for {num_prediction_points} time points...")

with torch.no_grad(): # Disable gradient calculations
    for i in tqdm(range(0, last_possible_start_index + 1, PREDICTION_STRIDE), desc="Predicting"): # Iterate up to the last possible start
        # Extract input sequence
        input_sequence_df = df_full_scaled.iloc[i : i + SEQUENCE_LENGTH]
        input_sequence_np = input_sequence_df.values # Includes scaled target

        # Convert to tensor and predict
        input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        scaled_prediction = model(input_tensor) # Shape: (1, FORECAST_HORIZON)
        scaled_prediction_np = scaled_prediction.cpu().numpy() # Shape: (1, FORECAST_HORIZON)

        # --- CLAMPING: Ensure scaled predictions are not negative ---
        scaled_prediction_np = np.maximum(0, scaled_prediction_np)
        # Optionally, if you know your scaler maps to [0,1] and want to be strict:
        # scaled_prediction_np = np.clip(scaled_prediction_np, 0, 1)

        # Inverse transform
        scaled_prediction_np_reshaped = scaled_prediction_np.reshape(FORECAST_HORIZON, 1)
        try:
            actual_prediction = target_scaler.inverse_transform(scaled_prediction_np_reshaped) # Shape: (FORECAST_HORIZON, 1)
        except ValueError as e:
            print(f"\nError during inverse_transform at step i={i}: {e}")
            print(f"Shape passed to inverse_transform: {scaled_prediction_np_reshaped.shape}")
            print(f"Scaler target n_features_in_: {target_scaler.n_features_in_}")
            raise e

        actual_prediction_flat = actual_prediction.flatten() # Shape: (FORECAST_HORIZON,)

        # Store as dictionary {step: value}
        # Step indices are 0-based (0 to FORECAST_HORIZON-1)
        prediction_dict = {step: float(value) for step, value in enumerate(actual_prediction_flat)}

        # Optional: Add timestamp of the *start* of the input sequence for context
        # prediction_dict['input_start_time'] = input_sequence_df.index[0].isoformat()

        all_predictions_list.append(prediction_dict)

print(f"\n--- Prediction Complete. Generated {len(all_predictions_list)} forecasts. ---")

# --- Save Predictions to JSON ---
print("\n--- Saving Predictions to JSON ---")
# Make filename dynamic
predictions_save_path = JSON_PATH / "historical_predictions.json"
try:
    with open(predictions_save_path, 'w') as f:
        json.dump(all_predictions_list, f, indent=4)
    print(f"Saved historical predictions to {predictions_save_path}")
except Exception as e:
    print(f"Error saving predictions to JSON: {e}")

print("\n--- Historical Forecast Script Finished ---") 