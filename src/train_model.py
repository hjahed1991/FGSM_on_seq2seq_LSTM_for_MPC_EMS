# train_model.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import MinMaxScaler # Scalers are loaded
import matplotlib.pyplot as plt
import os
import random
import pathlib
import joblib
import json
import argparse # Added for command-line arguments
import sys
import re

# --- Determine script directory --- 
script_dir = pathlib.Path(__file__).parent.resolve()
print(f"Script directory: {script_dir}")

# --- Import Model Utils --- 
from model_utils import (
        select_meteo_features,
        select_temporal_features,
        create_time_features,
        create_sequences,
        TimeSeriesDataset,
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
args = parser.parse_args()
print(f"Arguments: {args}")

# --- Configuration from Arguments --- (NEW)
TARGET_COLUMN = args.target
MODEL_TYPE = args.model
FORECAST_HORIZON = args.horizon
SEQUENCE_LENGTH = args.sequence_length
TIMESTAMP_COLUMN = 'time' # Keep fixed
RANDOM_SEED = args.seed # Keep fixed for tuning reproducibility
PROCESS_TYPE = 'clean'

# Print effective configuration being used
print("\n--- Effective Configuration (from args) ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Process Type: {PROCESS_TYPE}")

# --- Define Paths Based on Config --- (NEW/Revised)
# Base path for experiments (in the script_dir)

EXPERIMENTS_BASE_PATH = script_dir / 'experiments'
# Specific path for this run's tuning artifacts
CLEAN_ARTIFACTS_SAVE_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / 'clean_training_artifacts'
print(f"Artifacts Save Path: {CLEAN_ARTIFACTS_SAVE_PATH}")

# Create artifacts directory if it doesn't exist
os.makedirs(CLEAN_ARTIFACTS_SAVE_PATH, exist_ok=True)


print("--- Starting Final Model Training ---")

# Update paths and other configs based on potentially modified variables (especially TARGET_COLUMN)
DATA_PATH = '../../data/dataset.csv' # Keep data path fixed for now, or adjust based on TARGET_COLUMN if needed
print(f"Data Path: {DATA_PATH}")

METEO_FEATURES = select_meteo_features(TARGET_COLUMN)

TEMPORAL_FEATURES = select_temporal_features(TARGET_COLUMN)
# Include past target value as an input feature
INPUT_FEATURES = METEO_FEATURES + TEMPORAL_FEATURES + [TARGET_COLUMN]
print(f"Input Features: {INPUT_FEATURES}")

TEST_DAYS = 16
VALIDATION_DAYS = 15 # For splitting train data into train/validation

# N_CV_SPLITS = 11      # No longer needed as we use the full train_val_df_scaled
MAX_EPOCHS_FINAL = 200
PATIENCE_FINAL = MAX_EPOCHS_FINAL # Set patience equal to max epochs to disable early stopping

MODEL_SAVE_PATH = CLEAN_ARTIFACTS_SAVE_PATH / 'models'
ARTIFACTS_SAVE_PATH = CLEAN_ARTIFACTS_SAVE_PATH / 'artifacts' #  save to this path
PLOTS_SAVE_PATH = CLEAN_ARTIFACTS_SAVE_PATH / 'plots'
EXCEL_SAVE_PATH = CLEAN_ARTIFACTS_SAVE_PATH / 'excel_files'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(ARTIFACTS_SAVE_PATH, exist_ok=True)
os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)
os.makedirs(EXCEL_SAVE_PATH, exist_ok=True)

# --- Save Training Configuration ---
print("\n--- Saving Training Configuration ---")
training_config = {
    'TARGET_COLUMN': TARGET_COLUMN,
    'TIMESTAMP_COLUMN': TIMESTAMP_COLUMN,
    'INPUT_FEATURES': INPUT_FEATURES,
    # 'TEMPORAL_FEATURES': TEMPORAL_FEATURES, # List of generated temporal features
    'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
    'FORECAST_HORIZON': FORECAST_HORIZON,
    'TEST_DAYS': TEST_DAYS,
    'MAX_EPOCHS_FINAL': MAX_EPOCHS_FINAL,
    'PATIENCE_FINAL': PATIENCE_FINAL,
    'RANDOM_SEED': RANDOM_SEED,
    'MODEL_TYPE': MODEL_TYPE,
    'PROCESS_TYPE': PROCESS_TYPE,
    # Add any other relevant config if needed later
}
train_config_save_path = ARTIFACTS_SAVE_PATH / "training_config.json"
try:
    with open(train_config_save_path, 'w') as f:
        json.dump(training_config, f, indent=4)
    print(f"Training configuration saved to {train_config_save_path}")
except Exception as e:
    print(f"Error saving training configuration: {e}")
    # Decide if we should exit or just warn
    # exit()

# Set random seeds for reproducibility
print(f"\n--- Seeding with Effective Random Seed: {RANDOM_SEED} ---")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED)

# Set environment variable for deterministic CuBLAS *early*
if DEVICE == 'cuda':
    print("Setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic GPU operations (early).")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

print(f"Using device: {DEVICE}")

ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / 'tuning_artifacts'

# --- Load Hyperparameters and Scalers ---
print("\n--- Loading Hyperparameters and Scalers ---")

# Load hyperparameters
hyperparams_path = ARTIFACTS_LOAD_PATH / "best_hyperparams.json"
try:
    with open(hyperparams_path, 'r') as f:
        best_hyperparams = json.load(f)
    print(f"Loaded best hyperparameters from {hyperparams_path}:")
    print(best_hyperparams)
except FileNotFoundError:
    print(f"Error: Hyperparameter file not found at {hyperparams_path}. Run tune_hyperparameters.py first.")
    exit()

# Load scalers
feature_scaler_path = ARTIFACTS_LOAD_PATH / "feature_scaler.joblib"
target_scaler_path = ARTIFACTS_LOAD_PATH / "target_scaler.joblib"
try:
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print(f"Loaded feature scaler from {feature_scaler_path}")
    print(f"Loaded target scaler from {target_scaler_path}")
except FileNotFoundError:
    print(f"Error: Scaler file not found. Run tune_hyperparameters.py first.")
    exit()

# --- Load and Preprocess Full Data (using loaded scalers) ---
print("\n--- Loading and Preprocessing Data ---")

df = pd.read_csv(DATA_PATH,
                 parse_dates=['time'],
                 index_col=['time']) # Load with parsing/indexing
print(f"Loaded dataset with shape: {df.shape}")

# Generate time features
time_features_df = create_time_features(df.index, TARGET_COLUMN)
df = pd.concat([df, time_features_df], axis=1)
print(f"Added temporal features. Shape: {df.shape}")

# Handle missing values
if df.isnull().sum().any():
    print("Missing values found. Applying forward fill.")
    df = df.ffill()
else:
    print("No missing values found.")

# Select relevant columns
print(f"\n--- Selecting only needed Columns --- ")
df = df[INPUT_FEATURES]
print(f"DataFrame shape after selecting columns: {df.shape}")

# Split data
if not isinstance(df.index, pd.DatetimeIndex):
    raise TypeError("Index must be DatetimeIndex.")
test_split_date = df.index.max() - pd.Timedelta(days=TEST_DAYS)
train_val_df = df[df.index <= test_split_date].copy()
test_df = df[df.index > test_split_date].copy()

# Split train_val_df into training and validation sets BEFORE scaling
val_split_date_from_train = train_val_df.index.max() - pd.Timedelta(days=VALIDATION_DAYS)
val_df = train_val_df[train_val_df.index > val_split_date_from_train].copy()
train_df = train_val_df[train_val_df.index <= val_split_date_from_train].copy()


print(f"Train data shape (unscaled): {train_df.shape}")
print(f"Validation data shape (unscaled): {val_df.shape}")
print(f"Test data shape (unscaled): {test_df.shape}")

# Scale data using LOADED scalers
print("\n--- Scaling Data using Loaded Scalers ---")
# Note: Scalers were fit on the combined train_val set in the tuning script.
# We apply the same transform here.
scaled_features_train = feature_scaler.transform(train_df[INPUT_FEATURES])
scaled_features_val = feature_scaler.transform(val_df[INPUT_FEATURES])
scaled_features_test = feature_scaler.transform(test_df[INPUT_FEATURES])

scaled_target_train = target_scaler.transform(train_df[[TARGET_COLUMN]])
scaled_target_val = target_scaler.transform(val_df[[TARGET_COLUMN]])
scaled_target_test = target_scaler.transform(test_df[[TARGET_COLUMN]])


# Create scaled DataFrames (important for consistent data structure)
train_df_scaled = pd.DataFrame(scaled_features_train,
                                     columns=INPUT_FEATURES,
                                     index=train_df.index)
train_df_scaled[TARGET_COLUMN] = scaled_target_train

val_df_scaled = pd.DataFrame(scaled_features_val,
                                columns=INPUT_FEATURES,
                                index=val_df.index)
val_df_scaled[TARGET_COLUMN] = scaled_target_val


test_df_scaled = pd.DataFrame(scaled_features_test,
                                columns=INPUT_FEATURES,
                                index=test_df.index)
test_df_scaled[TARGET_COLUMN] = scaled_target_test

print("Scaled DataFrames created.")
print(f"Scaled Train shape: {train_df_scaled.shape}")
print(f"Scaled Validation shape: {val_df_scaled.shape}")
print(f"Scaled Test shape: {test_df_scaled.shape}")

# --- Prepare Final Training/Validation/Test Sets ---
print("\n--- Preparing Final Datasets from Scaled Data ---")

# Create sequences for train, validation, and test sets using SCALED data
X_train_final, y_train_final = create_sequences(
    train_df_scaled[INPUT_FEATURES].values,
    train_df_scaled[[TARGET_COLUMN]].values,
    SEQUENCE_LENGTH, FORECAST_HORIZON
)
X_val_final, y_val_final = create_sequences(
    val_df_scaled[INPUT_FEATURES].values,
    val_df_scaled[[TARGET_COLUMN]].values,
    SEQUENCE_LENGTH, FORECAST_HORIZON
)

X_test, y_test = create_sequences( # Create test sequences from test_df_scaled
    test_df_scaled[INPUT_FEATURES].values,
    test_df_scaled[[TARGET_COLUMN]].values,
    SEQUENCE_LENGTH, FORECAST_HORIZON
)

final_train_dataset = TimeSeriesDataset(X_train_final, y_train_final)
final_val_dataset = TimeSeriesDataset(X_val_final, y_val_final)

# test_dataset = TimeSeriesDataset(X_test, y_test) # Not needed for training

print(f"Created final train dataset with {len(final_train_dataset)} sequences.")
print(f"Created final validation dataset with {len(final_val_dataset)} sequences.")
print(f"Created test sequences: X shape {X_test.shape}, y shape {y_test.shape}")

# Use the batch size from loaded hyperparameters
final_batch_size = best_hyperparams['batch_size']

final_train_loader = DataLoader(final_train_dataset, batch_size=final_batch_size, shuffle=True, num_workers=0)
final_val_loader = DataLoader(final_val_dataset, batch_size=final_batch_size, shuffle=False, num_workers=0)

# --- Callback to collect losses ---
class LossHistory(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = [] # Re-add val_losses

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Check if 'train_loss_epoch' or 'train_loss' is available
        train_loss_key = None
        if 'train_loss' in trainer.callback_metrics:
            train_loss_key = 'train_loss'
        elif 'train_loss_epoch' in trainer.callback_metrics: # Common for epoch-end logging
            train_loss_key = 'train_loss_epoch'

        if train_loss_key and trainer.callback_metrics[train_loss_key] is not None:
            self.train_losses.append(trainer.callback_metrics[train_loss_key].item())
        elif hasattr(pl_module, 'current_train_loss') and pl_module.current_train_loss is not None: # Fallback for manually logged loss
             self.train_losses.append(pl_module.current_train_loss)
        # else:
            # Optionally log a warning if train loss is not found
            # print("Warning: Could not retrieve train_loss from callback_metrics or pl_module.")


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Do not log sanity check validation loss
        if trainer.sanity_checking:
            return
        # Check if 'val_loss' is available in callback_metrics
        if 'val_loss' in trainer.callback_metrics and trainer.callback_metrics['val_loss'] is not None:
            self.val_losses.append(trainer.callback_metrics['val_loss'].item())
        # else:
            # Optionally log a warning if validation loss is not found
            # print("Warning: Could not retrieve val_loss from callback_metrics.")


# --- Initialize Model with Best Hyperparameters ---
print("\n--- Initializing Model ---")

# Prepare parameters for model instantiation - Filter based on MODEL_TYPE
model_init_params = {
    'input_size': len(INPUT_FEATURES),
    'output_size': FORECAST_HORIZON,
    # Start with common params expected by all models (adjust if needed)
    'learning_rate': best_hyperparams['learning_rate'],
    'dropout_prob': best_hyperparams['dropout_prob']
}

# Select and initialize the correct model
try:
    if MODEL_TYPE == 'LSTM':
        # Add LSTM-specific params
        model_init_params['hidden_size'] = best_hyperparams['hidden_size']
        model_init_params['num_layers'] = best_hyperparams['num_layers']
        # Ensure teacher_forcing_ratio from Optuna is used for LSTMForecastModel2
        model_init_params['teacher_forcing_ratio'] = best_hyperparams.get('teacher_forcing_ratio', 0.5) # Default if missing, though tune_hyperparameters should save it
        final_model = LSTMForecastModel2(**model_init_params)
    elif MODEL_TYPE == 'GRU':
        # Add GRU-specific params
        model_init_params['hidden_size'] = best_hyperparams['hidden_size']
        model_init_params['num_layers'] = best_hyperparams['num_layers']
        model_init_params['teacher_forcing_ratio'] = best_hyperparams.get('teacher_forcing_ratio', 0.5) # Default if missing
        final_model = GRUForecastModel2(**model_init_params)
    elif MODEL_TYPE == 'Transformer':
        # Add Transformer-specific params
        model_init_params['d_model'] = best_hyperparams['d_model']
        model_init_params['nhead'] = best_hyperparams['nhead']
        model_init_params['num_encoder_layers'] = best_hyperparams['num_encoder_layers']
        model_init_params['num_decoder_layers'] = best_hyperparams['num_decoder_layers']
        model_init_params['dim_feedforward'] = best_hyperparams['dim_feedforward']
        model_init_params['src_max_seq_len'] = SEQUENCE_LENGTH
        model_init_params['teacher_forcing_ratio'] = best_hyperparams.get('teacher_forcing_ratio', 0.5)
        final_model = TransformerForecastModel(**model_init_params)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE for final training: {MODEL_TYPE}")

    print(f"Initialized {MODEL_TYPE} model with loaded hyperparameters:")
    print(final_model.hparams)

except KeyError as e:
    print(f"Error: Missing hyperparameter key '{e}' in loaded best_hyperparams.json for model type {MODEL_TYPE}.")
    print(f"Loaded params: {best_hyperparams}")
    exit()
except Exception as e:
    print(f"Error initializing model {MODEL_TYPE}: {e}")
    exit()

# --- Set Up Callbacks and Logger for Final Training ---
print("\n--- Setting Up Callbacks and Logger ---")

# Make filename dynamic
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', # Monitor validation loss
    dirpath=MODEL_SAVE_PATH,
    filename=f'best-model-{PROCESS_TYPE}-{MODEL_TYPE}-horizon_{FORECAST_HORIZON}-seed_{RANDOM_SEED}-{{epoch:02d}}-{{val_loss:.4f}}', # Filename reflects validation loss
    save_top_k=5, mode='min', verbose=True # Save top 5 models
)
final_early_stopping = EarlyStopping(
    monitor='val_loss', patience=PATIENCE_FINAL, verbose=True, mode='min' # Monitor validation loss
)
loss_history_callback = LossHistory()
final_logger = TensorBoardLogger(save_dir=CLEAN_ARTIFACTS_SAVE_PATH, name="lightning_logs") # Saves logs relative to script

# --- Train Final Model ---
print("\n--- Training Final Model ---")

final_trainer = pl.Trainer(
    max_epochs=MAX_EPOCHS_FINAL,
    accelerator=DEVICE,
    devices=1 if DEVICE != 'cpu' else 'auto',
    callbacks=[checkpoint_callback, final_early_stopping, loss_history_callback],
    logger=final_logger,
    enable_checkpointing=True,
    enable_progress_bar=True,
    deterministic=True
)

print("Starting final training run...")
final_trainer.fit(final_model, train_dataloaders=final_train_loader, val_dataloaders=final_val_loader) # Add validation dataloader

# --- Custom Model Selection Logic ---
print("\n--- Applying Custom Model Selection Logic ---")

# Get top 5 models from callback
top_k_models = checkpoint_callback.best_k_models
best_model_path = checkpoint_callback.best_model_path # Fallback

if not top_k_models or len(top_k_models) < 5:
    print(f"Warning: Did not find top 5 models. Found {len(top_k_models)}. Will use the best available model based on validation loss.")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("Error: No models were saved. Exiting.")
    exit()
else:
    print(f"Found {len(top_k_models)} models to choose from for custom selection.")
    
    model_performance = []
    
    # Extract epoch and validation loss from file paths and get corresponding train loss
    for path, val_loss in top_k_models.items():
        try:
            # Extract epoch number from filename
            epoch_str_match = re.search(r'epoch=(\d+)', path)
            if epoch_str_match:
                epoch = int(epoch_str_match.group(1))
                if epoch < len(loss_history_callback.train_losses):
                    train_loss = loss_history_callback.train_losses[epoch]
                    model_performance.append({
                        'epoch': epoch,
                        'path': path,
                        'val_loss': val_loss.item(), # Convert tensor to float
                        'train_loss': train_loss
                    })
                    # Define best_model_info here to ensure it's available later
                    if path == checkpoint_callback.best_model_path:
                        best_model_info_fallback = model_performance[-1]
                else:
                    print(f"Warning: Epoch {epoch} from path {path} out of range for train_losses.")
            else:
                print(f"Warning: Could not extract epoch from model path: {path}")
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not process model path {path}: {e}")

    if not model_performance:
        print("Error: Could not determine performance for any saved models. Using best_model_path as fallback.")
        best_model_path = checkpoint_callback.best_model_path
    else:
        # Sort by training loss to find the best among the top 5 val_loss models
        best_model_info = sorted(model_performance, key=lambda x: x['train_loss'])[0]
        best_model_path = best_model_info['path']
        
        print("\n--- Model Selection Summary ---")
        # Sort by validation loss for display purposes
        for model in sorted(model_performance, key=lambda x: x['val_loss']):
            selection_marker = "<- SELECTED" if model['path'] == best_model_path else ""
            print(f" - Epoch {model['epoch']}: Val Loss={model['val_loss']:.6f}, Train Loss={model['train_loss']:.6f} {selection_marker}")

        # Clean up other models, keeping only the selected best one
        for model_info_item in model_performance:
            if model_info_item['path'] != best_model_path:
                try:
                    if os.path.exists(model_info_item['path']):
                        os.remove(model_info_item['path'])
                        print(f"Removed surplus model: {model_info_item['path']}")
                except OSError as e:
                    print(f"Error removing model {model_info_item['path']}: {e}")


print(f"\nFinal training finished. Best clean model saved at: {best_model_path}")

# --- Plot Training Curves ---
print("\n--- Plotting and Saving Training Curves ---")

train_loss_history = loss_history_callback.train_losses
val_loss_history = loss_history_callback.val_losses

plt.figure(figsize=(12, 7))

# Plot training and validation loss
epochs_ran = range(len(train_loss_history))
plt.plot(epochs_ran, train_loss_history, label='Training Loss', marker='o', linestyle='-', markersize=5)
if val_loss_history:
    # Ensure validation history is not longer than training history for plotting
    plt.plot(epochs_ran[:len(val_loss_history)], val_loss_history, label='Validation Loss', marker='x', linestyle='--', markersize=5)

# Find and plot the best epoch based on the custom selection criteria
if 'best_model_info' in locals():
    best_epoch = best_model_info['epoch']
    min_val_loss = best_model_info['val_loss']
    corresponding_train_loss = best_model_info['train_loss']
    
    plt.scatter(best_epoch, min_val_loss, color='green', s=120, zorder=5, 
                label=f'Selected Epoch ({best_epoch}) Val Loss: {min_val_loss:.4f}')
    plt.scatter(best_epoch, corresponding_train_loss, color='red', s=120, zorder=5, 
                label=f'Selected Epoch ({best_epoch}) Train Loss: {corresponding_train_loss:.4f}')
    plt.axvline(x=best_epoch, color='grey', linestyle=':', linewidth=1, label='Selected epoch line')
    print(f"Selected model from Epoch {best_epoch} with Validation Loss: {min_val_loss:.6f} and Training Loss: {corresponding_train_loss:.6f}")

elif val_loss_history and not all(np.isnan(v) for v in val_loss_history): # Fallback to min val loss if custom logic failed
    min_val_loss = np.nanmin(val_loss_history)
    min_val_epoch_index = np.nanargmin(val_loss_history)
    min_val_epoch = epochs_ran[min_val_epoch_index]
    
    plt.scatter(min_val_epoch, min_val_loss, color='red', s=120, zorder=5,
                label=f'Best Epoch ({min_val_epoch}) by Val Loss: {min_val_loss:.4f}')
    plt.axvline(x=min_val_epoch, color='grey', linestyle=':', linewidth=1, label='Best epoch line')
    print(f"Minimum Validation Loss: {min_val_loss:.6f} at Epoch {min_val_epoch}")
else:
    print("Could not determine the best epoch from loss history.")


plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title(f"Training & Validation Loss Curve for {TARGET_COLUMN}-{PROCESS_TYPE}-{MODEL_TYPE}_model-horizon_{FORECAST_HORIZON}-seed_{RANDOM_SEED}") # Updated title
plt.legend()
plt.grid(True)
plot_save_path = PLOTS_SAVE_PATH / "training_loss_curve.png"
plt.savefig(plot_save_path)
print(f"Saved training loss plot to {plot_save_path}")
plt.close() # Close the plot figure

# --- Save Loss Data to Excel ---
print("Saving loss data...")
loss_data_save_dir = EXCEL_SAVE_PATH # Save in excel_files folder
# os.makedirs(loss_data_save_dir, exist_ok=True) # ARTIFACTS_PATH should already exist

try:
    # Use the same loss history lists used for plotting
    num_epochs = len(train_loss_history)
    epochs = list(range(num_epochs))

    # Pad validation loss if its length is less than training loss
    padded_val_loss = val_loss_history + [None] * (num_epochs - len(val_loss_history))

    loss_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': train_loss_history,
        'Validation Loss': padded_val_loss
    })

    # Add best epoch info if it was calculated (based on custom criteria)
    if 'best_model_info' in locals():
        best_epoch = best_model_info['epoch']
        loss_df['Is Selected Best Epoch'] = loss_df['Epoch'] == best_epoch
    else:
        loss_df['Is Selected Best Epoch'] = False

    excel_filename = 'training_loss_history.xlsx'
    excel_save_path = loss_data_save_dir / excel_filename
    loss_df.to_excel(excel_save_path, index=False, engine='openpyxl')
    print(f"Saved loss data to {excel_save_path}")

except Exception as e:
    print(f"Error saving loss data to Excel: {e}")


# --- Save Test Set Artifacts for Evaluation ---
print("\n--- Saving Test Set Artifacts for Evaluation ---")

# Save the test data sequences (already created above)
test_X_path = ARTIFACTS_SAVE_PATH / "test_X.npy"
test_y_path = ARTIFACTS_SAVE_PATH / "test_y.npy"
np.save(test_X_path, X_test)
np.save(test_y_path, y_test)
print(f"Saved test X sequences to {test_X_path}")
print(f"Saved test y sequences to {test_y_path}")

# Save timestamps for the test predictions
timestamp_start_index = SEQUENCE_LENGTH
timestamp_end_index = len(test_df) - FORECAST_HORIZON + 1
if len(y_test) != (timestamp_end_index - timestamp_start_index):
    print(f"Warning: Mismatch between y_test length ({len(y_test)}) and timestamp range calculation. Adjusting.")
    test_timestamps = test_df.index[timestamp_start_index : timestamp_start_index + len(y_test)]
else:
    test_timestamps = test_df.index[timestamp_start_index : timestamp_end_index]

test_timestamps_path = ARTIFACTS_SAVE_PATH / "test_timestamps.npy"
np.save(test_timestamps_path, test_timestamps.to_numpy())
print(f"Saved test timestamps to {test_timestamps_path}")

print("\nTraining script finished. Run evaluate_model.py to see test set predictions.") 