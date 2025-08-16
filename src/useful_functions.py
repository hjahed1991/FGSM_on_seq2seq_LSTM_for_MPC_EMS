# In the name of God
# All useful functions will be stored in this file

# Imports:
import os
import pandas as pd
from datetime import datetime


def load_data(file_path):
    """
    Loading time-series data from csv file and store it in a pandas dataframe.
    """
    data = pd.read_csv(file_path, parse_dates=['time'], index_col=['time'])
    return data


def round_value(value, digits=2):
    """
    This function will round a value to a given number of digits.
    Args:
        value: the value to round.
        digits: number of digits to round.

    Returns:rounded value.

    """
    return round(value, digits) if abs(value) > 1e-2 else 0

def train_val_test_split(df,
                         val_start="2021-01-01 00:00:00",
                         test_start="2021-01-15 00:00:00"):
    """
    This function will split the dataset into three separate
    train, validation and test sets, based on 2 specific date that
    gets from function parameters (val_start and test_start)
    """
    train = df.loc[df.index < val_start]
    val = df.loc[(df.index >= val_start) & (df.index < test_start)]
    test = df.loc[df.index >= test_start]

    return train, val, test

