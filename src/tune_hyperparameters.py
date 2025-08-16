# Tune Hyperparameters for Model

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import optuna.visualization as vis # Added for Optuna plots
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import os
import random
import pathlib # Added import
import joblib # Added import
import json # Added for saving hyperparameters
import argparse # Added for command-line arguments
import sys

# --- Determine experiment directory and Import Model Utils --- (NEW)
# Get the directory of the current script
script_dir = pathlib.Path(__file__).parent.resolve()
print(f"Script directory: {script_dir}")

from model_utils import (
        select_meteo_features,
        select_temporal_features,
        create_time_features,
        create_sequences,
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
args = parser.parse_args()
print(f"Arguments: {args}")

# --- Configuration from Arguments --- (NEW)
TARGET_COLUMN = args.target
MODEL_TYPE = args.model
FORECAST_HORIZON = args.horizon
SEQUENCE_LENGTH = args.sequence_length
TIMESTAMP_COLUMN = 'time' # Keep fixed
RANDOM_SEED = 42 # Keep fixed for tuning reproducibility


# Print effective configuration being used
print("\n--- Effective Configuration (from args) ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Sequence Length: {SEQUENCE_LENGTH}")
print(f"Random Seed: {RANDOM_SEED}")

# --- Define Paths Based on Config --- (NEW/Revised)
# Base path for experiments (in the script_dir)

EXPERIMENTS_BASE_PATH = script_dir / 'experiments'
# Specific path for this run's tuning artifacts
ARTIFACTS_SAVE_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / 'tuning_artifacts'
print(f"Artifacts Save Path: {ARTIFACTS_SAVE_PATH}")

# Create artifacts directory if it doesn't exist
os.makedirs(ARTIFACTS_SAVE_PATH, exist_ok=True)

# Update paths and other configs based on potentially modified variables (especially TARGET_COLUMN)
DATA_PATH = '../../data/dataset.csv' # Keep data path fixed for now, or adjust based on TARGET_COLUMN if needed
print(f"Data Path: {DATA_PATH}")

METEO_FEATURES = select_meteo_features(TARGET_COLUMN)

TEMPORAL_FEATURES = select_temporal_features(TARGET_COLUMN)

# Include past target value as an input feature
INPUT_FEATURES = METEO_FEATURES + TEMPORAL_FEATURES + [TARGET_COLUMN]
print(f"Input Features: {INPUT_FEATURES}")

TEST_DAYS = 16       # Number of days for the test set
OPTUNA_TRIALS = 30   # Number of Optuna trials
N_CV_SPLITS = 11   # Number of splits for TimeSeriesSplit
MAX_EPOCHS_OPTUNA = 10 # Max epochs during Optuna trials
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# Set random seeds for reproducibility using the potentially updated RANDOM_SEED
print(f"\n--- Seeding with Effective Random Seed: {RANDOM_SEED} ---")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED) # Seed Lightning

# Set environment variable for deterministic CuBLAS *early*
# Needs to be done before CUDA initializes contexts that might check this
if DEVICE == 'cuda':
    print("Setting CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic GPU operations (early).")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # As an alternative, try torch.use_deterministic_algorithms(True) here or after imports
    # However, the environment variable is often the recommended way for CuBLAS issues.

print(f"Using device: {DEVICE}")
print(f"Input features ({len(INPUT_FEATURES)}): {INPUT_FEATURES}")

print("\n--- Loading Data ---")
df = pd.read_csv(DATA_PATH,
                 parse_dates=['time'],
                 index_col=['time'])

print(f"Loaded dataset with shape: {df.shape}")

# --- Preprocessing Data ---
print("\n--- Preprocessing Data ---")

# Generate and add time features
time_features_df = create_time_features(df.index, TARGET_COLUMN)
df = pd.concat([df, time_features_df], axis=1)
print(f"Added temporal features. DataFrame shape now: {df.shape}")

# Check for missing values
if df.isnull().sum().any():
    print("Missing values found. Applying forward fill.")
    df = df.ffill()
else:
    print("No missing values found.")

# Ensure all input features exist
missing_features = [f for f in INPUT_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"Missing required features in dataset: {missing_features}")

# Separating features and Target from the original dataset

print(f"\n--- Selecting only needed Columns for Model (Features + Target)---")
print(f"Target variable: {TARGET_COLUMN}")
print(f"Feature columns ({len(INPUT_FEATURES)}): {INPUT_FEATURES}")

df = df[INPUT_FEATURES]
print(f"DataFrame shape after selecting needed columns: {df.shape}")
# Split data into train/validation and test sets
# Assuming hourly data, calculate split point
if not isinstance(df.index, pd.DatetimeIndex):
     raise TypeError("Index must be DatetimeIndex for time-based splitting.")

test_split_date = df.index.max() - pd.Timedelta(days=TEST_DAYS)
train_val_df = df[df.index <= test_split_date].copy()
test_df = df[df.index > test_split_date].copy()

print(f"Train/Validation data shape: {train_val_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Train/Validation period: {train_val_df.index.min()} to {train_val_df.index.max()}")
print(f"Test period: {test_df.index.min()} to {test_df.index.max()}")


# Scaling data (fit on train_val, transform train_val and test)
print("\n--- Scaling Data ---")

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

print("Fitting scalers on train/val data...")
feature_scaler.fit(train_val_df[INPUT_FEATURES])
target_scaler.fit(train_val_df[[TARGET_COLUMN]])

print("Transforming train/val and test data...")
# Transform train/val and test data for features
scaled_features_train_val = feature_scaler.transform(train_val_df[INPUT_FEATURES])
scaled_features_test = feature_scaler.transform(test_df[INPUT_FEATURES])
# Transform target data for train/val and test
scaled_target_train_val = target_scaler.transform(train_val_df[[TARGET_COLUMN]])
scaled_target_test = target_scaler.transform(test_df[[TARGET_COLUMN]])

train_val_df_scaled = pd.DataFrame(scaled_features_train_val,
                                    columns=INPUT_FEATURES,
                                      index=train_val_df.index)

train_val_df_scaled[TARGET_COLUMN] = scaled_target_train_val

test_df_scaled = pd.DataFrame(scaled_features_test,
                                    columns=INPUT_FEATURES,
                                      index=test_df.index)

test_df_scaled[TARGET_COLUMN] = scaled_target_test

print("Scaled DataFrames created.")

print("\nSample of scaled training data:")
print(train_val_df_scaled.head())
print("\nCheck min/max values in scaled training data (should be approx 0 and 1):")
print(train_val_df_scaled.agg(['min', 'max']))


# --- Defining Optuna Objective ---
print("\n--- Defining Optuna Objective ---")

def objective(trial: optuna.trial.Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""

    # --- Suggest Hyperparameters Conditionally ---
    # Common HPs
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_prob = trial.suggest_float("dropout_prob", 0.1, 0.5)
    # Add teacher_forcing_ratio for LSTMForecastModel2
    teacher_forcing_ratio = trial.suggest_float("teacher_forcing_ratio", 0.0, 1.0)

    model_params = {
        'input_size': len(INPUT_FEATURES),
        'output_size': FORECAST_HORIZON,
        'dropout_prob': dropout_prob,
        'learning_rate': learning_rate,
        'teacher_forcing_ratio': teacher_forcing_ratio # Add to model_params
    }

    if MODEL_TYPE in ['LSTM', 'GRU']:
        model_params['hidden_size'] = trial.suggest_int("hidden_size", 32, 128, step=16)
        model_params['num_layers'] = trial.suggest_int("num_layers", 1, 3)
    elif MODEL_TYPE == 'Transformer':
        # Suggest d_model first, needs to be divisible by nhead
        # Powers of 2 are common, ensuring divisibility by common nhead values (2, 4, 8)
        d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])

        # Prune trial if d_model is not divisible by nhead
        if d_model % nhead != 0:
            print(f"Pruning trial {trial.number}: d_model ({d_model}) not divisible by nhead ({nhead}).")
            raise optuna.exceptions.TrialPruned()

        model_params['d_model'] = d_model
        model_params['nhead'] = nhead
        model_params['num_encoder_layers'] = trial.suggest_int("num_encoder_layers", 1, 4)
        model_params['num_decoder_layers'] = trial.suggest_int("num_decoder_layers", 1, 4)
        model_params['dim_feedforward'] = trial.suggest_int("dim_feedforward", 128, 512, step=64)
        model_params['src_max_seq_len'] = SEQUENCE_LENGTH
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

    # --- Cross-validation Loop ---
    # Use the SCALED DataFrame for Optuna
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    val_losses = []

    fold_count = 0
    # Use train_val_df_scaled here
    for train_indices, val_indices in tscv.split(train_val_df_scaled):
        fold_count += 1
        print(f"\n--- Optuna Trial {trial.number}, Fold {fold_count}/{N_CV_SPLITS} ---")
        train_fold_df = train_val_df_scaled.iloc[train_indices]
        val_fold_df = train_val_df_scaled.iloc[val_indices]

        # Create sequences for this fold using scaled data
        X_train_fold, y_train_fold = create_sequences(
            train_fold_df[INPUT_FEATURES].values,
            train_fold_df[[TARGET_COLUMN]].values,
            SEQUENCE_LENGTH,
            FORECAST_HORIZON
        )
        X_val_fold, y_val_fold = create_sequences(
            val_fold_df[INPUT_FEATURES].values,
            val_fold_df[[TARGET_COLUMN]].values,
            SEQUENCE_LENGTH,
            FORECAST_HORIZON
        )

        # Use imported TimeSeriesDataset
        train_fold_dataset = TimeSeriesDataset(X_train_fold, y_train_fold)
        val_fold_dataset = TimeSeriesDataset(X_val_fold, y_val_fold)

        train_loader = DataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_fold_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # --- Select and Initialize Model ---
        try:
            if MODEL_TYPE == 'LSTM':
                model = LSTMForecastModel2(**model_params)
            elif MODEL_TYPE == 'GRU':
                model = GRUForecastModel2(**model_params)
            elif MODEL_TYPE == 'Transformer':
                model = TransformerForecastModel(**model_params)
            else:
                # This case should be caught earlier, but added for safety
                 raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
        except Exception as e:
             print(f"Error initializing model {MODEL_TYPE} with params {model_params}: {e}")
             return float('inf') # Return high loss if model init fails

        # Callbacks
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=False, mode="min")
        # pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss") # Still removed

        # Trainer
        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS_OPTUNA,
            accelerator=DEVICE,
            devices=1 if DEVICE != 'cpu' else 'auto',
            callbacks=[early_stop_callback],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        try:
            # Train the model
            trainer.fit(model, train_loader, val_loader)

            # Get best validation loss
            best_val_loss = early_stop_callback.best_score.item()
            if best_val_loss is None or np.isnan(best_val_loss) or best_val_loss == float('inf'):
                print(f"Warning: Invalid validation loss ...")
                return float('inf')
            val_losses.append(best_val_loss)

            # Pruning logic would go here if enabled

        except Exception as e:
            print(f"Error during training fold ...: {e}")
            return float('inf')

    # Calculate average validation loss
    average_val_loss = np.mean(val_losses)
    print(f"--- Optuna Trial {trial.number} Completed ...")

    return average_val_loss

# --- Running Optuna Study ---
print("\n--- Running Optuna Study ---")

# Create study
study = optuna.create_study(direction='minimize',
                            pruner=optuna.pruners.MedianPruner())

# Start optimization
try:
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=None, n_jobs=1)
except KeyboardInterrupt:
    print("Optuna study interrupted by user.")

# Get best trial
best_trial = study.best_trial

print(f"\n--- Optuna Study Finished ---")
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial value (average validation loss): {best_trial.value:.6f}")
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

best_hyperparams = best_trial.params


# --- Save Best Hyperparameters and Scalers ---
print("\n--- Saving Best Hyperparameters and Scalers ---")

# Save hyperparameters
hyperparams_path = ARTIFACTS_SAVE_PATH / "best_hyperparams.json"
try:
    os.remove(hyperparams_path)
    print(f"Removed existing hyperparameters file: {hyperparams_path}")
except FileNotFoundError:
    pass
with open(hyperparams_path, 'w') as f:
    json.dump(best_hyperparams, f, indent=4)
print(f"Saved best hyperparameters to {hyperparams_path}")

# Save the scalers
feature_scaler_path = ARTIFACTS_SAVE_PATH / "feature_scaler.joblib"
target_scaler_path = ARTIFACTS_SAVE_PATH / "target_scaler.joblib"
try:
    os.remove(feature_scaler_path)
    os.remove(target_scaler_path)
    print(f"Removed existing scaler files in {ARTIFACTS_SAVE_PATH}")
except FileNotFoundError:
    pass
# Save the *actual fitted scaler objects* from the scaling section
joblib.dump(feature_scaler, feature_scaler_path)
joblib.dump(target_scaler, target_scaler_path)
print(f"Saved feature scaler to {feature_scaler_path}")
print(f"Saved target scaler to {target_scaler_path}")

print("\nHyperparameter tuning finished. Run train_model.py to train the final model.")
