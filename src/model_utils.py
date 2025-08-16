# lstm_model_utils.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
import math # Added for PositionalEncoding
import random

def select_meteo_features(TARGET_COLUMN: str) -> list[str]:
    if TARGET_COLUMN == 'consumption':
        return ['sun_azimuth:d', 'wind_speed_100m:ms', 'dew_point_10m:C', 't_100m:C']
    elif TARGET_COLUMN == 'wind_production':
        return ['wind_speed_100m:ms', 'dew_point_100m:C',]
    elif TARGET_COLUMN == 'pv_production':
        return ['global_rad_1h:Wh', 'sun_elevation:d', 'sunshine_duration_1h:min', 'temp', 'effective_cloud_cover:p', 'relative_humidity_2m:p',]
    else:
        return []

def select_temporal_features(TARGET_COLUMN: str) -> list[str]:
    if TARGET_COLUMN == 'consumption':
        return ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos', 'is_weekend', 'is_morning_peak', 'is_evening_peak']
    elif TARGET_COLUMN == 'wind_production':
        return ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
    elif TARGET_COLUMN == 'pv_production':
        return ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
    else:
        return []

# --- Time Feature Engineering Function ---
def create_time_features(df_index: pd.DatetimeIndex, target_column_name: str) -> pd.DataFrame:
    """
    Creates advanced time features based on the input Datetime index.
    Conditionally adds features based on the target column name.

    Input:
        df_index: Index of type pd.DatetimeIndex
        target_column_name: The name of the target column being predicted.

    Output: DataFrame containing new time features
    """
    # Direct use of index for better performance
    month = df_index.month
    day_of_year = df_index.dayofyear
    day_of_week = df_index.dayofweek
    hour = df_index.hour

    df_features = pd.DataFrame(index=df_index) # Create DataFrame with the same index

    # Hour as sinusoidal and cosinusoidal (to preserve the 24-hour cyclical nature)
    df_features['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df_features['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

    # Day of the week as sinusoidal and cosinusoidal (to preserve the 7-day cyclical nature)
    df_features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

    # Day of the year as sinusoidal and cosinusoidal (to preserve the 365-day cyclical nature)
    # Using 365.25 to account for leap years (optional, minimal impact)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Month as sinusoidal and cosinusoidal (to preserve the 12-month cyclical nature)
    df_features['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df_features['month_cos'] = np.cos(2 * np.pi * month / 12.0)

    # Add consumption-specific features only if target is 'consumption'
    if target_column_name == 'consumption':
        # Is it weekend? (Saturday=5, Sunday=6 in dayofweek)
        df_features['is_weekend'] = (day_of_week >= 5).astype(float)

        # Is it morning peak hours (e.g., 7 to 10)?
        df_features['is_morning_peak'] = ((hour >= 7) & (hour <= 10)).astype(float)

        # Is it evening peak hours (e.g., 17 to 20)?
        df_features['is_evening_peak'] = ((hour >= 17) & (hour <= 20)).astype(float)

    return df_features


def features_to_poison(TARGET_COLUMN: str) -> list[str]:
    if TARGET_COLUMN == 'consumption':
        return ['sun_azimuth:d', 'wind_speed_100m:ms', 'dew_point_10m:C', 't_100m:C']
    elif TARGET_COLUMN == 'wind_production':
        return ['wind_speed_100m:ms', 'dew_point_100m:C',]
    elif TARGET_COLUMN == 'pv_production':
        return ['global_rad_1h:Wh', 'sun_elevation:d', 'sunshine_duration_1h:min', 'temp', 'effective_cloud_cover:p', 'relative_humidity_2m:p',]


# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose pe to (1, max_len, d_model) to match batch_first=True format for broadcasting.
        self.register_buffer('pe', pe.transpose(0, 1)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is (batch, seq_len, d_model)
        # self.pe is (1, max_len, d_model). We slice it to match seq_len
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Sequence Creation Function ---
def create_sequences(input_data: np.ndarray, target_data: np.ndarray, seq_length: int, forecast_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Creates sequences of data for time series forecasting.

    Args:
        input_data: Array of input features (including target potentially) [n_samples, n_features].
        target_data: Array of target variable [n_samples, 1].
        seq_length: Length of the input sequence (history).
        forecast_horizon: Length of the output sequence (future predictions).

    Returns:
        A tuple containing:
         - np.ndarray: Input sequences (X) of shape [num_sequences, seq_length, n_features].
         - np.ndarray: Output sequences (y) of shape [num_sequences, forecast_horizon].
    """
    X, y = [], []
    n_samples = len(input_data)
    for i in range(n_samples - seq_length - forecast_horizon + 1):
        X.append(input_data[i : i + seq_length])
        y.append(target_data[i + seq_length : i + seq_length + forecast_horizon].squeeze()) # Squeeze to remove last dim if 1
    return np.array(X), np.array(y)

# --- PyTorch Dataset Class ---
class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series sequences."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# --- seq2seq LSTM Model Definition ---
class LSTMForecastModel2(pl.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout_prob: float, learning_rate: float,
                 teacher_forcing_ratio: float = 0.5): # Added teacher_forcing_ratio
        super().__init__()
        self.save_hyperparameters() # Saves args to self.hparams
        self.output_size = output_size # Forecast horizon
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Input shape: (batch, seq_len, features)
            dropout=dropout_prob if num_layers > 1 else 0
        )

        # --- Decoder ---
        # Decoder input size will be 1 (the previous predicted value)
        self.decoder_lstm = nn.LSTM(
            input_size=1, # Takes the previous output step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.decoder_dropout = nn.Dropout(dropout_prob)
        # Linear layer maps decoder hidden state to a single output step
        self.decoder_linear = nn.Linear(hidden_size, 1) # Predicts one step at a time
        # Apply ReLU to ensure non-negative outputs
        # self.output_activation = nn.ReLU()

        self.criterion = nn.MSELoss() # Mean Squared Error Loss

    def forward(self, x, y=None): # y is needed for teacher forcing
        # x shape: (batch, seq_len, input_size)
        # y shape: (batch, output_size) [Optional, for teacher forcing]

        batch_size = x.size(0)

        # --- Encoder ---
        # We only need the final hidden and cell states from the encoder
        _, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)
        # encoder_hidden shape: (num_layers, batch, hidden_size)
        # encoder_cell shape: (num_layers, batch, hidden_size)

        # --- Decoder ---
        # Initialize decoder input (start token): Use the last value of the input target
        # or a zero tensor if target isn't part of input 'x'.
        # Assuming the target is the *first* feature in x for simplicity:
        # decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(2) # (batch, 1, 1)
        # Alternative: Start with zeros
        decoder_input = torch.zeros(batch_size, 1, 1, device=x.device) # (batch, 1, 1)


        # Initialize decoder hidden state with encoder's final hidden state
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        predictions = []

        for t in range(self.output_size):
            # decoder_input shape: (batch, 1, 1)
            # decoder_hidden shape: (num_layers, batch, hidden_size)
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                decoder_input, (decoder_hidden, decoder_cell)
            )
            # decoder_output shape: (batch, 1, hidden_size)

            # Apply dropout and linear layer to the decoder output
            decoder_output_dropped = self.decoder_dropout(decoder_output.squeeze(1)) # (batch, hidden_size)
            prediction_step = self.decoder_linear(decoder_output_dropped) # (batch, 1)
            # Apply activation per step to ensure non-negativity
            # prediction_step = self.output_activation(prediction_step)

            predictions.append(prediction_step.unsqueeze(1)) # Store prediction (batch, 1, 1)

            # Decide whether to use teacher forcing for the next input
            use_teacher_force = (y is not None) and (random.random() < self.teacher_forcing_ratio) and self.training
            if use_teacher_force:
                # Teacher forcing: Use actual target value as the next input
                decoder_input = y[:, t].unsqueeze(1).unsqueeze(2) # Shape: (batch, 1, 1)
            else:
                # No teacher forcing: Use the model's own prediction as the next input
                decoder_input = prediction_step.unsqueeze(2) # Shape: (batch, 1, 1)

        # Concatenate predictions along the time dimension
        predictions_tensor = torch.cat(predictions, dim=1) # Shape: (batch, output_size, 1)

        # Squeeze the last dimension
        return predictions_tensor.squeeze(-1) # Shape: (batch, output_size)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, y) # Pass target y for teacher forcing
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Don't use teacher forcing for validation
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) # Don't use teacher forcing for testing
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


