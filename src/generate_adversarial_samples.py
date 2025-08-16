# This file is used for generating adversarial samples 

import torch
import numpy as np
import pandas as pd
import joblib
import json
import os
import pytorch_lightning as pl
import pathlib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
import matplotlib.pyplot as plt
import random
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
        create_sequences,
        TimeSeriesDataset,
        LSTMForecastModel2,
        features_to_poison
        
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


# Script specific checks
if PROCESS_TYPE != 'poisoning':
    print(f"Error: This script requires 'process_type' to be 'poisoning', but found '{PROCESS_TYPE}'.")
    exit()
if ATTACK_TYPE != 'FGSM':
     print(f"Warning: Attack type from path is '{ATTACK_TYPE}', but this script implements FGSM. Ensure this is intended.")
     # Or exit()

EXPERIMENTS_BASE_PATH = script_dir / 'experiments'
# Specific path for this run's tuning artifacts
CLEAN_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / 'clean_training_artifacts'
print(f"Clean Artifacts load Path: {CLEAN_ARTIFACTS_LOAD_PATH}")

# Define paths based on params
# Load data/artifacts from the corresponding clean run
DATA_PATH = '../../data/dataset.csv' # Keep data path fixed for now, or adjust based on TARGET_COLUMN if needed
print(f"Data Path: {DATA_PATH}")

MODEL_LOAD_PATH = CLEAN_ARTIFACTS_LOAD_PATH / 'models' # Load clean model checkpoint directory
ARTIFACTS_LOAD_PATH = CLEAN_ARTIFACTS_LOAD_PATH / 'artifacts' # Load clean artifacts (scalers, train_config)
POISONED_DATA_SAVE_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / f'seed_{RANDOM_SEED}' / f'{PROCESS_TYPE}' / f'{ATTACK_TYPE}' / f'epsilon_{EPSILON}' / f'ratio_{POISONING_RATIO}'

os.makedirs(POISONED_DATA_SAVE_PATH, exist_ok=True)

UPPER_ARTIFACTS_LOAD_PATH = EXPERIMENTS_BASE_PATH / TARGET_COLUMN / MODEL_TYPE / f'horizon_{FORECAST_HORIZON}' / f'sequence_length_{SEQUENCE_LENGTH}' / 'tuning_artifacts' #  load from this path

#os.makedirs(POISONED_DATA_SAVE_PATH, exist_ok=True)

print(f"--- Starting {ATTACK_TYPE} Adversarial Sample Generation ({TARGET_COLUMN}/{MODEL_TYPE}/h{FORECAST_HORIZON}/seed{RANDOM_SEED}/eps{EPSILON}/ratio{POISONING_RATIO}) ---")

# FGSM parameters (Mostly from path now)
# FEATURES_TO_POISON remains hardcoded for now, could be made dynamic if needed
FEATURES_TO_POISON = features_to_poison(TARGET_COLUMN)
POISONING_TARGET_SET = 'train' # Target set to poison
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE_FGSM = 32 # Batch size for FGSM gradient calculation
PLOT_SAMPLE_COUNT = 64 # Number of poisoned samples to visualize per feature

# Print effective configuration being used
print("\n--- Effective FGSM Configuration ---")
print(f"Target Series: {TARGET_COLUMN}")
print(f"Model Type: {MODEL_TYPE}")
print(f"Forecast Horizon: {FORECAST_HORIZON}")
print(f"Random Seed: {RANDOM_SEED}")
print(f"Process Type: {PROCESS_TYPE}")
print(f"Attack Type: {ATTACK_TYPE}")
print(f"Epsilon: {EPSILON}")
print(f"Poisoning Ratio: {POISONING_RATIO}")
print(f"Features to Poison: {FEATURES_TO_POISON}")
print(f"Data Load Path: {DATA_PATH}")
print(f"Clean Model Load Path: {MODEL_LOAD_PATH}")
print(f"Clean Artifacts Load Path: {ARTIFACTS_LOAD_PATH}")
print(f"Poisoned Data Save Path: {POISONED_DATA_SAVE_PATH}")
print(f"Device: {DEVICE}")

# Load training config from the CLEAN run artifacts
print(f"\n--- Loading Training Config from Clean Run ({ARTIFACTS_LOAD_PATH}) ---")
train_config_path = ARTIFACTS_LOAD_PATH / "training_config.json"
try:
    with open(train_config_path, 'r') as f:
        train_config = json.load(f)
    print(f"Loaded clean training configuration from {train_config_path}")
    TIMESTAMP_COLUMN = train_config.get('TIMESTAMP_COLUMN', 'time')
    INPUT_FEATURES = train_config['INPUT_FEATURES'] # Use features clean model trained on
    SEQUENCE_LENGTH = train_config['SEQUENCE_LENGTH']
    TEST_DAYS = train_config['TEST_DAYS']
    # Verify consistency (optional)
    if train_config.get('TARGET_COLUMN') != TARGET_COLUMN:
        print(f"Warning: TARGET_COLUMN mismatch: path={TARGET_COLUMN}, config={train_config.get('TARGET_COLUMN')}")
    if train_config.get('FORECAST_HORIZON') != FORECAST_HORIZON:
         print(f"Warning: FORECAST_HORIZON mismatch: path={FORECAST_HORIZON}, config={train_config.get('FORECAST_HORIZON')}")
    if train_config.get('MODEL_TYPE') != MODEL_TYPE:
         print(f"Warning: MODEL_TYPE mismatch: path={MODEL_TYPE}, config={train_config.get('MODEL_TYPE')}")

except FileNotFoundError:
    print(f"Error: Clean training config file not found at {train_config_path}.")
    exit()
except KeyError as e:
    print(f"Error: Missing key {e} in clean training_config.json.")
    exit()

print(f"Using Input Features from clean run: {INPUT_FEATURES}")

# Set random seeds for reproducibility using the potentially updated RANDOM_SEED
print(f"\n--- Seeding with Effective Random Seed: {RANDOM_SEED} ---")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
pl.seed_everything(RANDOM_SEED) # Seed Lightning


# --- Load Clean Model (Dynamic) ---
print(f"--- Loading Clean {MODEL_TYPE} Model from ({MODEL_LOAD_PATH}) ---")
try:
    checkpoint_files = list(MODEL_LOAD_PATH.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No CLEAN checkpoint files found in {MODEL_LOAD_PATH}")
    best_clean_model_path = max(checkpoint_files, key=os.path.getctime)
    print(f"Found best clean model checkpoint: {best_clean_model_path}")

    # Load dynamically
    if MODEL_TYPE == 'LSTM':
        clean_model = LSTMForecastModel2.load_from_checkpoint(best_clean_model_path, map_location=torch.device(DEVICE))
    elif MODEL_TYPE == 'GRU':
        clean_model = GRUForecastModel2.load_from_checkpoint(best_clean_model_path, map_location=torch.device(DEVICE))
    elif MODEL_TYPE == 'Transformer':
        clean_model = TransformerForecastModel.load_from_checkpoint(best_clean_model_path, map_location=torch.device(DEVICE))
    else:
        raise ValueError(f"Unsupported MODEL_TYPE for loading: {MODEL_TYPE}")

    clean_model.to(DEVICE)
    clean_model.eval()
    print(f"Clean {MODEL_TYPE} model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Run train_model.py for the clean case first.")
    exit()
except Exception as e:
    print(f"Error loading clean {MODEL_TYPE} model from {best_clean_model_path}: {e}")
    exit()

# --- Load Scalers from Clean Run ---
print(f"--- Loading Scalers from Clean Run ({UPPER_ARTIFACTS_LOAD_PATH}) ---")
feature_scaler_path = UPPER_ARTIFACTS_LOAD_PATH / "feature_scaler.joblib"
target_scaler_path = UPPER_ARTIFACTS_LOAD_PATH / "target_scaler.joblib"
try:
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    # Use feature names from scaler if available, otherwise from train_config
    scaler_features = list(feature_scaler.feature_names_in_) if hasattr(feature_scaler, 'feature_names_in_') else INPUT_FEATURES
    if len(scaler_features) != feature_scaler.n_features_in_:
         warnings.warn(f"Mismatch between derived scaler_features count ({len(scaler_features)}) and scaler expected features ({feature_scaler.n_features_in_}). Check consistency.")

    print(f"Loaded feature scaler from {feature_scaler_path} (expects {feature_scaler.n_features_in_} features)")
    print(f"Loaded target scaler from {target_scaler_path}")
except FileNotFoundError:
    print(f"Error: Scaler file not found in {ARTIFACTS_LOAD_PATH}. Run tune/train for the clean case first.")
    exit()
except Exception as e:
    print(f"Error loading scalers: {e}")
    exit()


# --- Load and Preprocess Original Data ---
print("--- Loading and Preprocessing Original Data ---")
try:
    df = pd.read_csv(DATA_PATH,
                     parse_dates=[TIMESTAMP_COLUMN],
                     index_col=TIMESTAMP_COLUMN)
    print(f"Loaded original dataset '{os.path.basename(DATA_PATH)}' with shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Original data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error reading data file: {e}")
    exit()

# Generate time features (using dynamic TARGET_COLUMN)
time_features_df = create_time_features(df.index, TARGET_COLUMN)
df = pd.concat([df, time_features_df], axis=1)
print(f"Added temporal features. Shape: {df.shape}")

# Get the list of temporal features actually generated (needed for cleanup)
TEMPORAL_FEATURES_GENERATED = list(time_features_df.columns)

# Handle missing values
if df.isnull().sum().any():
    print("Missing values found. Applying forward fill.")
    df = df.ffill()
    if df.isnull().sum().any():
        print("Warning: Missing values remain after ffill. Filling with 0.")
        df = df.fillna(0) # Fallback
else:
    print("No missing values found.")

# Select relevant columns (ensure order matches scaler)
# Use the features expected by the *scaler* (scaler_features derived above)
try:
    # Check if all scaler_features are present in df
    missing_scaler_feats = [f for f in scaler_features if f not in df.columns]
    if missing_scaler_feats:
        raise KeyError(f"Features expected by scaler but missing from DataFrame: {missing_scaler_feats}")

    df_processed = df[scaler_features].copy()
    print(f"Selected relevant features for scaling using scaler's expected features. Shape: {df_processed.shape}")
except KeyError as e:
    print(f"Error selecting features for scaling: {e}")
    print(f"Scaler expects: {scaler_features}")
    print(f"DataFrame has: {list(df.columns)}")
    exit()


# --- Split Data (Indices) ---
print("--- Splitting Data into Train/Test Sets ---")
if not isinstance(df_processed.index, pd.DatetimeIndex):
    raise TypeError("Index must be DatetimeIndex for splitting.")
test_split_date = df_processed.index.max() - pd.Timedelta(days=TEST_DAYS)
train_indices = df_processed.index <= test_split_date
test_indices = df_processed.index > test_split_date

df_train_val = df_processed[train_indices].copy()
df_test = df_processed[test_indices].copy()

print(f"Train/Validation data shape (unscaled): {df_train_val.shape}")
print(f"Test data shape (unscaled): {df_test.shape}")


# --- Scale Data ---
print("--- Scaling Data using Loaded Scalers ---")
# Important: Scale train/val and test sets separately but using the same scaler instance
scaled_features_train_val = feature_scaler.transform(df_train_val[scaler_features])
scaled_features_test = feature_scaler.transform(df_test[scaler_features])
scaled_target_train_val = target_scaler.transform(df_train_val[[TARGET_COLUMN]])
scaled_target_test = target_scaler.transform(df_test[[TARGET_COLUMN]])
# Create scaled DataFrames
df_train_val_scaled = pd.DataFrame(scaled_features_train_val,
                                     columns=scaler_features,
                                     index=df_train_val.index)
df_train_val_scaled[TARGET_COLUMN] = scaled_target_train_val
df_test_scaled = pd.DataFrame(scaled_features_test,
                                 columns=scaler_features,
                                 index=df_test.index)
df_test_scaled[TARGET_COLUMN] = scaled_target_test

print("Data scaled successfully.")
print(f"Scaled Train/Val shape: {df_train_val_scaled.shape}")
print(f"Scaled Test shape: {df_test_scaled.shape}")

print("\nSample of scaled training data:")
print(df_train_val_scaled.head())
print("\nCheck min/max values in scaled training data (should be approx 0 and 1):")
print(df_train_val_scaled.agg(['min', 'max']))

# --- Identify Poisoning Targets ---
print("--- Identifying Poisoning Targets in Training Data ---")
if POISONING_TARGET_SET != 'train':
    raise NotImplementedError("Currently only poisoning the 'train' set is supported.")

num_train_samples = len(df_train_val_scaled)
num_poisoned = int(num_train_samples * POISONING_RATIO)
if num_poisoned == 0 and POISONING_RATIO > 0:
     warnings.warn(f"Poisoning ratio {POISONING_RATIO} resulted in 0 samples to poison for training set size {num_train_samples}. No poisoning will occur.")
elif num_poisoned > num_train_samples:
     num_poisoned = num_train_samples # Cap at total samples
     warnings.warn(f"Poisoning ratio {POISONING_RATIO} too high. Poisoning all {num_train_samples} training samples.")

# Select random indices from the *training DataFrame* to poison
poison_indices_df = np.random.choice(df_train_val_scaled.index, size=num_poisoned, replace=False)
print(f"Randomly selected {num_poisoned} ({POISONING_RATIO*100}%) training samples (indices) to poison.")

# Determine feature indices to poison
feature_names = scaler_features # Use the list of features from the scaler/config
try:
    if FEATURES_TO_POISON == 'all':
        features_to_poison_names = feature_names
    elif isinstance(FEATURES_TO_POISON, list):
        features_to_poison_names = [f for f in FEATURES_TO_POISON if f in feature_names]
        if len(features_to_poison_names) != len(FEATURES_TO_POISON):
            warnings.warn("Some features specified in FEATURES_TO_POISON were not found in the data/scaler features.")
    else:
        raise ValueError("Invalid value for FEATURES_TO_POISON. Use 'all' or a list of feature names.")

    poison_feature_indices = [feature_names.index(f) for f in features_to_poison_names]
    print(f"Will poison {len(poison_feature_indices)} features: {features_to_poison_names}")
    if not poison_feature_indices:
        print("Warning: No valid features selected for poisoning based on FEATURES_TO_POISON.")
        num_poisoned = 0 # No features means no poisoning
except ValueError as e:
    print(f"Error processing FEATURES_TO_POISON: {e}")
    exit()
except Exception as e:
     print(f"An unexpected error occurred determining features to poison: {e}")
     exit()


# ---  Perturbation Generation (Adapted for DataFrame Rows) ---
print(f"--- Generating {ATTACK_TYPE} {PROCESS_TYPE} Perturbations on Scaled Training Data ---")

# Create a copy to store poisoned data
df_train_val_poisoned_scaled = df_train_val_scaled.copy()

if num_poisoned > 0 and poison_feature_indices:
    # Define CPU device
    cpu_device = torch.device('cpu')
    print(f"\nMoving model to CPU ({cpu_device}) for {ATTACK_TYPE} gradient calculation to avoid cuDNN issues...")
    model_on_cpu = clean_model.to(cpu_device)
    model_on_cpu.eval() # Ensure CPU model starts in eval mode

    # Convert relevant parts of the DataFrame to numpy for easier slicing
    scaled_train_val_data = df_train_val_scaled[feature_names].values
    scaled_train_val_target = df_train_val_scaled[[TARGET_COLUMN]].values

    # Get the integer indices corresponding to the selected datetime indices
    all_train_indices_list = list(df_train_val_scaled.index)
    try:
        poison_indices_iloc = [all_train_indices_list.index(ts) for ts in poison_indices_df]
    except ValueError as e:
        print(f"Error finding timestamp index during poisoning setup: {e}. This might indicate issues with index alignment.")
        # Restore cuDNN status if it was changed
        # torch.backends.cudnn.enabled = cudnn_enabled_status
        exit()

    print(f"Applying {ATTACK_TYPE} perturbations (gradients computed on CPU)...")
    for i in tqdm(poison_indices_iloc, desc="Applying FGSM to selected samples (CPU grad)"):
        # To poison the row at index `i`, we need the sequence ENDING just BEFORE `i`
        # to predict the timestep that INCLUDES `i` (depending on FORECAST_HORIZON).
        # OR, more simply, consider the sequence that *includes* the data point at index `i`.
        # Let's perturb the sequence *ending* at index `i`. The perturbation at `i`
        # will be based on the gradient calculated using the sequence `i-SEQUENCE_LENGTH+1` to `i`.

        seq_start_idx = i - SEQUENCE_LENGTH + 1
        seq_end_idx = i + 1
        if seq_start_idx < 0: continue
        x_seq = scaled_train_val_data[seq_start_idx:seq_end_idx, :]
        y_seq_start_idx = i + 1
        y_seq_end_idx = i + 1 + FORECAST_HORIZON
        if y_seq_end_idx > len(scaled_train_val_target): continue
        y_seq = scaled_train_val_target[y_seq_start_idx:y_seq_end_idx]
        y_seq = y_seq.flatten()

        # Convert to tensors ON CPU
        # Use .copy() to avoid warnings about non-writable numpy arrays
        x_tensor_cpu = torch.FloatTensor(x_seq.copy()).unsqueeze(0).to(cpu_device)
        y_tensor_cpu = torch.FloatTensor(y_seq.copy()).unsqueeze(0).to(cpu_device)
        x_tensor_cpu.requires_grad = True

        # --- FGSM Calculation on CPU ---
        # Forward pass on CPU model
        outputs_cpu = model_on_cpu(x_tensor_cpu)
        loss = model_on_cpu.criterion(outputs_cpu, y_tensor_cpu)

        # Zero gradients on CPU model
        model_on_cpu.zero_grad()

        # Temporarily switch CPU model to train() mode for backward pass
        model_on_cpu.train()
        try:
            loss.backward()
        except Exception as e:
            print(f"\nError during loss.backward() on CPU for index {i}: {e}")
            model_on_cpu.eval() # Ensure model is back in eval mode even after error
            continue # Skip this sample
        # Switch back to eval() mode immediately
        model_on_cpu.eval()

        if x_tensor_cpu.grad is None:
            print(f"Warning: Gradient is None for index {i} (CPU calculation). Skipping perturbation.")
            continue

        data_grad_cpu = x_tensor_cpu.grad.data

        # Get the sign of the gradient (on CPU)
        sign_data_grad_cpu = data_grad_cpu.sign()

        # Create perturbation for the *entire sequence* (on CPU)
        perturbed_x_seq_tensor_cpu = x_tensor_cpu + EPSILON * sign_data_grad_cpu
        # Clamp the entire perturbed sequence (on CPU)
        perturbed_x_seq_tensor_cpu = torch.clamp(perturbed_x_seq_tensor_cpu, 0, 1)

        # Extract the perturbed values for the target features at the specific timestep `i` (on CPU)
        perturbed_values_at_i_cpu = perturbed_x_seq_tensor_cpu[0, -1, poison_feature_indices].detach()

        # --- Update the DataFrame --- (using CPU tensor -> numpy)
        target_timestamp = df_train_val_scaled.index[i]
        target_feature_names = [feature_names[idx] for idx in poison_feature_indices]
        df_train_val_poisoned_scaled.loc[target_timestamp, target_feature_names] = perturbed_values_at_i_cpu.numpy()

        # Detach and delete CPU tensors to free memory
        x_tensor_cpu.requires_grad = False
        del x_tensor_cpu, y_tensor_cpu, outputs_cpu, loss, data_grad_cpu, sign_data_grad_cpu, perturbed_x_seq_tensor_cpu, perturbed_values_at_i_cpu
        # No need for torch.cuda.empty_cache() here as computations are on CPU

    print(f"{ATTACK_TYPE} perturbations applied.")
    # Move model back to original device if needed elsewhere (optional, as original clean_model is untouched)
    # clean_model.to(DEVICE)
    # No need to restore cuDNN status as it wasn't changed here

    # --- Visualize Poisoned vs Original Samples --- 
    print("\n--- Visualizing Perturbations for Selected Features ---")
    plot_save_dir = POISONED_DATA_SAVE_PATH / f"{ATTACK_TYPE}_{PROCESS_TYPE}_perturbation_plots"
    os.makedirs(plot_save_dir, exist_ok=True)
    print(f"Saving plots to: {plot_save_dir}")

    # Sort the poisoned indices (which are timestamps) for potentially clearer plotting
    sorted_poison_indices = np.sort(poison_indices_df)

    for feature_name in features_to_poison_names:
        try:
            # Select a subset of poisoned indices to plot
            if len(sorted_poison_indices) > PLOT_SAMPLE_COUNT:
                plot_indices = sorted_poison_indices[:PLOT_SAMPLE_COUNT]
                plot_title_suffix = f" (First {PLOT_SAMPLE_COUNT} Poisoned Samples)"
            else:
                plot_indices = sorted_poison_indices # Plot all if fewer than count
                plot_title_suffix = f" (All {len(sorted_poison_indices)} Poisoned Samples)"

            # Select original and poisoned values ONLY at the subset of timestamps
            original_vals = df_train_val_scaled.loc[plot_indices, feature_name]
            poisoned_vals = df_train_val_poisoned_scaled.loc[plot_indices, feature_name]
            timestamps = plot_indices # These are the actual datetime indices

            plt.figure(figsize=(15, 6))
            # Plotting sorted data, no need for extra sort here
            plt.plot(timestamps, original_vals, 'bo-', label='Original Scaled', markersize=4, alpha=0.7)
            plt.plot(timestamps, poisoned_vals, 'rx--', label='Poisoned Scaled', markersize=4, alpha=0.7)

            # Add grid to the plot
            plt.grid(True)

            plt.title(f"{ATTACK_TYPE} Perturbation: Feature '{feature_name}'\n(Epsilon={EPSILON}, Ratio={POISONING_RATIO}){plot_title_suffix}")
            plt.xlabel("Timestamp")
            plt.ylabel("Scaled Value (0-1)")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Sanitize feature name for filename
            safe_feature_name = feature_name.replace(":", "_").replace(" ", "_")
            plot_filename = f"perturbation_{safe_feature_name}.png"
            plot_path = plot_save_dir / plot_filename
            plt.savefig(plot_path)
            plt.close() # Close the figure to free memory
            # print(f"Saved plot for feature '{feature_name}' to {plot_path}")
        except Exception as e:
            print(f"Warning: Could not generate plot for feature '{feature_name}'. Error: {e}")
            if plt.fignum_exists(plt.gcf().number):
                 plt.close() # Ensure plot is closed even if error occurred mid-plot

else:
    print("Skipping FGSM perturbation generation (num_poisoned=0 or no features selected).")


# --- Combine Poisoned Train and Clean Test Data ---
print("--- Combining Poisoned Train and Clean Test Data ---")
df_poisoned_scaled = pd.concat([df_train_val_poisoned_scaled, df_test_scaled], axis=0)
print(f"Combined scaled dataframe shape: {df_poisoned_scaled.shape}")


# --- Inverse Transform Data ---
print("--- Inverse Transforming Data ---")
# Ensure the columns are in the exact order expected by the feature scaler
if not list(df_poisoned_scaled.columns) == scaler_features:
     print("Warning: Column order mismatch before inverse scaling. Reordering...")
     df_poisoned_scaled = df_poisoned_scaled[scaler_features]

# Inverse transform features
try:
    # Create a temporary array because inverse_transform modifies inplace if input is df
    scaled_data_array = df_poisoned_scaled[scaler_features].values
    inversed_features = feature_scaler.inverse_transform(scaled_data_array)
except ValueError as e:
    print(f"Error during inverse scaling: {e}")
    print("This might happen if the data shape or values are unexpected (e.g., NaNs introduced).")
    # Optional: Add debugging here, e.g., check df_poisoned_scaled for NaNs/Infs
    print(df_poisoned_scaled.isnull().sum())
    exit()


# Create DataFrame with inversed features
df_poisoned_inversed = pd.DataFrame(inversed_features,
                                      columns=scaler_features, # Use original feature names
                                      index=df_poisoned_scaled.index)

# Note: The target column was scaled along with features if it was part of INPUT_FEATURES
# and handled by feature_scaler. If target_scaler was used *separately* in train_model
# (e.g., on the target column *only* before creating sequences), this needs adjustment.
# Based on train_model.py structure review, it seems TARGET_COLUMN was included in INPUT_FEATURES
# and scaled by feature_scaler. So target_scaler might not be needed here if loaded feature_scaler covers all.
# If target_scaler *was* separate, we would need to inverse transform the target column separately here.
# Let's assume feature_scaler handled all INPUT_FEATURES including target.
# We can add a check:
if TARGET_COLUMN not in scaler_features:
    print(f"Warning: Target column '{TARGET_COLUMN}' not found in feature scaler columns. Cannot inverse transform target using feature scaler.")
    # If target scaler exists and was used separately:
    # target_col_scaled = df_poisoned_scaled[[TARGET_COLUMN]].values
    # target_col_inversed = target_scaler.inverse_transform(target_col_scaled)
    # df_poisoned_inversed[TARGET_COLUMN] = target_col_inversed
    # else: pass or raise error


print("Inverse scaling completed.")
print(df_poisoned_inversed.head())
print(df_poisoned_inversed.columns)

#--- Clean Up Columns and Save ---
print("--- Cleaning Columns and Saving Poisoned Dataset ---")

# Identify original columns (before adding temporal features)
original_columns = [col for col in df.columns if col not in TEMPORAL_FEATURES_GENERATED]
# Ensure TARGET_COLUMN is present if it was in the original data
if TARGET_COLUMN not in original_columns and TARGET_COLUMN in df_poisoned_inversed.columns:
     original_columns.append(TARGET_COLUMN) # Should already be there if part of df.columns

# Select only the original columns from the inversed data
try:
    df_final_poisoned = df_poisoned_inversed[original_columns].copy()
    print(f"Selected original columns. Final shape: {df_final_poisoned.shape}")
except KeyError as e:
    print(f"Error selecting original columns: {e}. Some original columns might be missing after processing.")
    print(f"Columns available after inverse transform: {list(df_poisoned_inversed.columns)}")
    print(f"Attempting to save with available columns that were originally present...")
    # Fallback: save columns that are both original and currently available
    available_original_cols = [col for col in original_columns if col in df_poisoned_inversed.columns]
    df_final_poisoned = df_poisoned_inversed[available_original_cols].copy()
print(df_final_poisoned.head())
print(df_final_poisoned.columns)

# Construct filename
csv_save_dir = POISONED_DATA_SAVE_PATH / f"poisoned_datasets_{ATTACK_TYPE}"
os.makedirs(csv_save_dir, exist_ok=True)

features_str = "_".join(features_to_poison_names) if features_to_poison_names else "none"
features_str = features_str.replace(":", "").replace(" ", "") # Sanitize feature names for filename
poisoned_filename = "poisoned_dataset.csv"
poisoned_save_path = csv_save_dir / poisoned_filename

# Save to CSV
df_final_poisoned.to_csv(poisoned_save_path)
print(f"Successfully saved poisoned dataset to: {poisoned_save_path}")

print("--- Adversarial Dataset Generation Finished ---") 