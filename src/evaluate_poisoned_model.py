# evaluate_poisoned_model.py

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
# Remove specific model import here, will import dynamically
# from model_utils import LSTMForecastModel, TimeSeriesDataset
from torch.utils.data import DataLoader
import pathlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Added imports
import math # Added import
import json # Added import
import argparse # Added for command-line arguments
import sys

# --- Determine experiment directory and Import Model Utils --- (NEW)
script_dir = pathlib.Path(__file__).parent.resolve()

print(f"Script directory: {script_dir}")

# --- Import Model Utils --- 
from model_utils import (
        # select_meteo_features, # Not strictly needed for eval if artifacts loaded
        # select_temporal_features, # Not strictly needed for eval if artifacts loaded
        # create_time_features, # Not strictly needed for eval if artifacts loaded
        # create_sequences, # Not strictly needed for eval if artifacts loaded
        TimeSeriesDataset,
        LSTMForecastModel2,
        GRUForecastModel2,
        TransformerForecastModel,
        
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


print("--- Starting Evaluation ---")

EXPERIMENTS_BASE_PATH = script_dir / 'experiments'
# Specific path for this run's tuning artifacts
CLEAN_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / 'clean_training_artifacts'
print(f"Clean Artifacts load Path: {CLEAN_ARTIFACTS_LOAD_PATH}")

POISONED_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / f'{PROCESS_TYPE}' / f'{ATTACK_TYPE}' / f'epsilon_{EPSILON}' / f'ratio_{POISONING_RATIO}'


# Define paths based on script location and extracted params
MODEL_LOAD_PATH = POISONED_LOAD_PATH / 'models' # Match train_model.py format
ARTIFACTS_LOAD_PATH = POISONED_LOAD_PATH / 'artifacts'
PLOTS_SAVE_PATH = POISONED_LOAD_PATH / 'plots'
EXCEL_SAVE_PATH = POISONED_LOAD_PATH / 'excel_files'

UPPER_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / 'tuning_artifacts' #  load from this path


# Check essential config
if TARGET_COLUMN == 'unknown' or MODEL_TYPE == 'unknown' or FORECAST_HORIZON is None or PROCESS_TYPE == 'unknown':
    print("Error: Essential configuration (target, model_type, horizon, process_type) could not be determined from path.")
    exit()

# Print effective configuration being used
print("\n--- Effective Evaluation Configuration ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Process Type: {PROCESS_TYPE}")
print(f"Seed (from path): {RANDOM_SEED}")
print(f"Model Load Path: {MODEL_LOAD_PATH}")
print(f"Artifacts Load Path: {ARTIFACTS_LOAD_PATH}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE_EVAL = 128 # Batch size for evaluation

# --- Load Artifacts ---
print("\n--- Loading Artifacts ---")

# Ensure directories exist
os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)
os.makedirs(EXCEL_SAVE_PATH, exist_ok=True)

# Load the target scaler
scaler_path = ARTIFACTS_LOAD_PATH / "target_scaler.joblib"
try:
    target_scaler = joblib.load(scaler_path)
    print(f"Loaded target scaler from {scaler_path}")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}. Run tune_hyperparameters.py first.")
    exit()

# Load the test data sequences
test_X_path = ARTIFACTS_LOAD_PATH / "test_X.npy"
test_y_path = ARTIFACTS_LOAD_PATH / "test_y.npy"
try:
    X_test = np.load(test_X_path)
    y_test = np.load(test_y_path)
    print(f"Loaded test X sequences from {test_X_path} (shape: {X_test.shape})")
    print(f"Loaded test y sequences from {test_y_path} (shape: {y_test.shape})")
except FileNotFoundError:
    print(f"Error: Test data file not found. Run train_model.py first.")
    exit()

# Load the test timestamps
test_timestamps_path = ARTIFACTS_LOAD_PATH / "test_timestamps.npy"
try:
    test_timestamps_np = np.load(test_timestamps_path)
    test_timestamps = pd.to_datetime(test_timestamps_np)
    print(f"Loaded test timestamps from {test_timestamps_path} (length: {len(test_timestamps)})")
except FileNotFoundError:
    print(f"Error: Test timestamps file not found. Run train_model.py first.")
    exit()

# --- Find and Load Best Model ---
print("\n--- Loading Best Model ---")

# Find the latest .ckpt file in the dynamic model directory
try:
    checkpoint_files = list(MODEL_LOAD_PATH.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {MODEL_LOAD_PATH}")
    # Assuming the best model is the last one saved by ModelCheckpoint (usually the case)
    best_model_path = max(checkpoint_files, key=os.path.getctime)
    print(f"Found best model checkpoint: {best_model_path}")
except FileNotFoundError as e:
    print(e)
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

    model.to(DEVICE) # Ensure model is on the correct device
    model.eval() # Set model to evaluation mode
    print(f"{MODEL_TYPE} model loaded successfully from {best_model_path} and set to evaluation mode.")
except Exception as e:
    print(f"Error loading {MODEL_TYPE} model from checkpoint {best_model_path}: {e}")
    exit()

# --- Create Test DataLoader ---
test_dataset_eval = TimeSeriesDataset(X_test, y_test)
test_loader_eval = DataLoader(test_dataset_eval, batch_size=BATCH_SIZE_EVAL, shuffle=False, num_workers=0)

# --- Make Predictions ---
print("\n--- Making Predictions on Test Set ---")
predictions_scaled = []
with torch.no_grad(): # Disable gradient calculations for inference
    for batch_x, _ in test_loader_eval:
        batch_x = batch_x.to(DEVICE)
        y_hat_batch = model(batch_x)
        predictions_scaled.append(y_hat_batch.cpu().numpy())

# Concatenate predictions from all batches
predictions_scaled = np.concatenate(predictions_scaled, axis=0)
print(f"Generated scaled predictions with shape: {predictions_scaled.shape}")

# --- CLAMPING: Ensure scaled predictions are not negative (consistent with clean_evaluate_model.py) ---
predictions_scaled = np.maximum(0, predictions_scaled)
# Optionally, if you know your scaler maps to [0,1] and want to be strict:
# predictions_scaled = np.clip(predictions_scaled, 0, 1)
print(f"Clamped scaled predictions to be non-negative. Shape: {predictions_scaled.shape}")

# Ensure predictions shape matches y_test shape
if predictions_scaled.shape != y_test.shape:
    try:
        predictions_scaled = predictions_scaled.reshape(y_test.shape)
        print(f"Reshaped predictions to: {predictions_scaled.shape}")
    except ValueError as e:
         print(f"Error: Shape mismatch between predictions {predictions_scaled.shape} and actuals {y_test.shape}. Cannot reshape. {e}")
         exit()

# --- Inverse Transform ---
print("\n--- Inverse Transforming Data ---")

original_shape = predictions_scaled.shape
num_sequences = original_shape[0]
forecast_horizon = original_shape[1]

# Reshape for inverse transform
predictions_reshaped = predictions_scaled.reshape(num_sequences * forecast_horizon, 1)
y_test_reshaped = y_test.reshape(num_sequences * forecast_horizon, 1)

# Inverse transform predictions and actuals
predictions_actual_scale_flat = target_scaler.inverse_transform(predictions_reshaped)
actuals_actual_scale_flat = target_scaler.inverse_transform(y_test_reshaped)

# Reshape back to original format
predictions_actual_scale = predictions_actual_scale_flat.reshape(original_shape)
actuals_actual_scale = actuals_actual_scale_flat.reshape(original_shape)

print(f"Inverse transformed predictions shape: {predictions_actual_scale.shape}")
print(f"Inverse transformed actuals shape: {actuals_actual_scale.shape}")

# --- Calculate Evaluation Metrics ---
print("\n--- Calculating Evaluation Metrics ---")

metrics = {}
all_actuals = []
all_preds = []
for i in range(FORECAST_HORIZON):
    step = i + 1
    actual_step = actuals_actual_scale[:, i]
    pred_step = predictions_actual_scale[:, i]

    # Store for overall calculation
    all_actuals.extend(actual_step)
    all_preds.extend(pred_step)

    mae = mean_absolute_error(actual_step, pred_step)
    mse = mean_squared_error(actual_step, pred_step)
    rmse = math.sqrt(mse)
    r2 = r2_score(actual_step, pred_step)

    metrics[f'Step_{step}_MAE'] = mae
    metrics[f'Step_{step}_RMSE'] = rmse
    metrics[f'Step_{step}_R2'] = r2

    print(f"Step {step}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# Calculate overall metrics across all steps
overall_mae = mean_absolute_error(all_actuals, all_preds)
overall_mse = mean_squared_error(all_actuals, all_preds)
overall_rmse = math.sqrt(overall_mse)
overall_r2 = r2_score(all_actuals, all_preds)

metrics['Overall_MAE'] = overall_mae
metrics['Overall_RMSE'] = overall_rmse
metrics['Overall_R2'] = overall_r2

print(f"\nOverall (All Steps): MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, R2={overall_r2:.4f}")
# Also report 1-step ahead for consistency
mae_step1 = metrics.get('Step_1_MAE', np.nan)
rmse_step1 = metrics.get('Step_1_RMSE', np.nan)
r2_step1 = metrics.get('Step_1_R2', np.nan)
print(f"Overall 1-Step Ahead: MAE={mae_step1:.4f}, RMSE={rmse_step1:.4f}, R2={r2_step1:.4f}")


# --- Save Metrics to Excel ---
metrics_df = pd.DataFrame([metrics]) # Create DataFrame from dictionary
# Make filename dynamic
excel_metrics_filename = 'test_set_evaluation_metrics.xlsx'
excel_metrics_path = EXCEL_SAVE_PATH / excel_metrics_filename
metrics_df.to_excel(excel_metrics_path, index=False)
print(f"\nSaved evaluation metrics to {excel_metrics_path}")

# --- Plotting Test Results ---
print("\n--- Plotting Test Results ---")

# Plotting 1-step ahead predictions
prediction_step_to_plot = 0 # Index 0 = 1st step ahead

plt.figure(figsize=(15, 7))
plt.plot(test_timestamps, actuals_actual_scale[:, prediction_step_to_plot], label='Actual', color='blue', marker='.', linestyle='-')
plt.plot(test_timestamps, predictions_actual_scale[:, prediction_step_to_plot], label='Predicted', color='red', marker='x', markersize=4, linestyle='--')

plt.xlabel("Time")
plt.ylabel(TARGET_COLUMN.replace('_', ' ').title())
title_step1 = f"{TARGET_COLUMN} - Test Set ({MODEL_TYPE}_model-horizon_{FORECAST_HORIZON}-seed_{RANDOM_SEED}-{ATTACK_TYPE}-{PROCESS_TYPE}-epsilon_{EPSILON}-ratio_{POISONING_RATIO}): Actual vs Predicted ({prediction_step_to_plot+1}-step ahead)\n"
title_step1 += f"MAE={mae_step1:.3f}, RMSE={rmse_step1:.3f}, R2={r2_step1:.3f}"
plt.title(title_step1)
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
# Make filename dynamic
plot_save_path_eval_step1 = PLOTS_SAVE_PATH / "test_set_predictions_vs_actuals_step1.png"
plt.savefig(plot_save_path_eval_step1)
print(f"Saved test prediction plot (step 1) to {plot_save_path_eval_step1}")
plt.close()

# --- Save 1-Step Prediction Data to Excel ---
print("Saving 1-step ahead prediction data to Excel...")
try:
    pred_df = pd.DataFrame({
        'Timestamp': test_timestamps,
        f'Actual_{TARGET_COLUMN}': actuals_actual_scale[:, prediction_step_to_plot],
        f'Predicted_{TARGET_COLUMN}_{MODEL_TYPE}_horizon_{FORECAST_HORIZON}_seed_{RANDOM_SEED}_{ATTACK_TYPE}_{PROCESS_TYPE}_epsilon_{EPSILON}_ratio_{POISONING_RATIO}': predictions_actual_scale[:, prediction_step_to_plot]
    })

    # Construct dynamic Excel filename
    excel_pred_filename = 'test_set_predictions_vs_actuals_step1.xlsx'
    excel_pred_path = EXCEL_SAVE_PATH / excel_pred_filename

    # Save to Excel
    pred_df.to_excel(excel_pred_path, index=False, engine='openpyxl')
    print(f"Saved prediction data to {excel_pred_path}")

except Exception as e:
    print(f"Error saving prediction data to Excel: {e}")

# --- Optional: Plotting a specific forecast horizon window ---
start_idx_plot = 100
num_steps_in_horizon = predictions_actual_scale.shape[1]

plt.figure(figsize=(15, 7))
forecast_start_index = start_idx_plot
# Handle potential NaT in timestamps if test set is small
if forecast_start_index >= len(test_timestamps):
    print(f"Warning: start_idx_plot ({start_idx_plot}) is out of bounds for test_timestamps (len: {len(test_timestamps)}). Skipping window plot.")
else:
    # Explicitly set frequency assuming hourly data ('H') or infer if possible
    freq = pd.infer_freq(test_timestamps)
    if freq is None:
        print("Warning: Could not infer frequency for forecast timestamps. Assuming hourly ('H').")
        freq = 'H'

    forecast_timestamps_window = pd.date_range(start=test_timestamps[forecast_start_index], periods=num_steps_in_horizon, freq=freq)

    plt.plot(forecast_timestamps_window,
             actuals_actual_scale[forecast_start_index, :],
             label='Actual', color='blue', marker='o', linestyle='-')
    plt.plot(forecast_timestamps_window,
             predictions_actual_scale[forecast_start_index, :],
             label=f'Forecast', color='red', marker='x', linestyle='--')

    plt.xlabel("Time")
    plt.ylabel(TARGET_COLUMN.replace('_', ' ').title())
    plt.title(f"Multi-Step Forecast vs Actual ({TARGET_COLUMN} - {MODEL_TYPE} - horizon_{FORECAST_HORIZON} - seed_{RANDOM_SEED} - {ATTACK_TYPE} - {PROCESS_TYPE} - epsilon_{EPSILON} - ratio_{POISONING_RATIO} ) - Starting index {forecast_start_index}")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Make filename dynamic
    plot_save_path_window = PLOTS_SAVE_PATH / f"Multistep_test_set_predictions_vs_actuals_window_start{forecast_start_index}.png"
    plt.savefig(plot_save_path_window)
    print(f"Saved test prediction window plot to {plot_save_path_window}")
    # plt.show() # Usually don't show in automated scripts
    plt.close()

print("\n--- Evaluation Finished ---") 