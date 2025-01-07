"""
Module for Data Fetching, Model Management, and Visualization

This module provides utility functions to:
1. Fetch historical OHLCV (Open, High, Low, Close, Volume) data from Binance US using the CCXT library.
2. Load and save datasets (CSV files) to/from local storage.
3. Save and load machine learning models using Keras.
4. Plot distribution of data with a Kernel Density Estimate (KDE) using Seaborn and Matplotlib.

Functions:
    - fetch_data(symbol, timeframe, start_date, end_date): Fetch historical OHLCV data from Binance US.
    - load_data(file_path): Load a CSV file into a pandas DataFrame.
    - save_data(data, file_path): Save a pandas DataFrame to a CSV file.
    - save_model(model, file_path): Save a machine learning model to a specified file path.
    - load_model(file_path): Load a machine learning model from a specified file path.
    - distribution_plot(data, bins=30): Plot the distribution of a given dataset with a KDE.

Dependencies:
    - ccxt: For interacting with the Binance US exchange API.
    - configparser: For reading the configuration file containing API keys and other settings.
    - pandas: For working with data in DataFrame format.
    - seaborn: For statistical data visualization.
    - matplotlib: For plotting the graphs.
    - tensorflow.keras: For saving and loading machine learning models.

Configuration:
    - The module requires a config.ini file containing API credentials for Binance US. The config file should have the following sections:
        [BINANCEUS]
        API_KEY = your_api_key
        SECRET = your_secret_key

    The config file should be located in the 'config' directory one level up from the current script.

Example Usage:
    1. Fetching data:
    >>> df = fetch_data('BTC/USDT', '1h', '2024-01-01 00:00:00', '2024-01-02 00:00:00')
    
    2. Loading and saving data:
    >>> data = load_data('data.csv')
    >>> save_data(data, 'new_data.csv')
    
    3. Saving and loading models:
    >>> save_model(model, 'model.keras')
    >>> model = load_model('model.keras')
    
    4. Plotting distribution:
    >>> distribution_plot(data['price'])

Notes:
    - The fetch_data function fetches data in batches, handling network errors and rate limits gracefully.
    - The model save/load functions assume compatibility with Keras models.
    - The distribution_plot function uses Seaborn's histplot with KDE for data visualization.
"""

import os
import time
import ccxt
import numpy as np
import configparser
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model as internal_load_model

# Set up the path to the config file
config_path = Path(__file__).resolve().parents[1] / 'config' / 'config.ini'

# Ensure the config file exists
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}.")

# Create a ConfigParser instance and read the config file
config = configparser.ConfigParser()
config.read(config_path)

def fetch_data(symbol, timeframe, start_date, end_date):
    """
    Fetch historical OHLCV (Open, High, Low, Close, Volume) data for a given symbol and timeframe from Binance US.

    Parameters:
    symbol (str): The trading pair symbol (e.g., 'BTC/USD').
    timeframe (str): The timeframe for the OHLCV data (e.g., '1m', '5m', '1h', '1d').
    start_date (str): The start date for fetching data in 'YYYY-MM-DD hh-mm-ss' format.
    end_date (str): The end date for fetching data in 'YYYY-MM-DD hh-mm-ss' format.

    Returns:
    pd.DataFrame: A DataFrame containing the OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """

    try:
        # Initialize the exchange with API credentials
        exchange = ccxt.binanceus({
            'apiKey': (config['BINANCEUS']['API_KEY']),
            'secret': (config['BINANCEUS']['SECRET']),
        })
    except Exception as e:
        print(f"Error initializing exchange: {e}")
        return pd.DataFrame()

    limit = 500  # Maximum number of data points to fetch in one request
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)  # Convert start date to timestamp in milliseconds
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)  # Convert end date to timestamp in milliseconds

    # Initialize an empty list to store all fetched data
    all_data = []
    
    # Fetch data in batches
    while start_timestamp < end_timestamp:
        try:
            # Fetch OHLCV data from the exchange
            ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=start_timestamp, limit=limit)
            
            if not ohlcv_data:
                break  # Exit loop if no data is returned

            # Convert the data to a DataFrame for easier date filtering
            batch_df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            batch_df['timestamp'] = pd.to_datetime(batch_df['timestamp'], unit='ms')

            # Filter out data that goes beyond the specified end_date
            batch_df = batch_df[batch_df['timestamp'] < pd.to_datetime(end_date)]
            
            # Check if batch_df is empty before accessing .iloc
            if not batch_df.empty:
                # Append filtered data to all_data list
                all_data.extend(batch_df.values)

                # Print the date range for the batch in human-readable format
                batch_start = batch_df['timestamp'].iloc[0]
                batch_end = batch_df['timestamp'].iloc[-1]
                print(f"Fetched data from {batch_start} to {batch_end}")

                # Update start_timestamp to the last fetched timestamp + 1 ms
                start_timestamp = int(batch_df['timestamp'].iloc[-1].timestamp() * 1000) + 1

                # Stop fetching if the last batch of data has reached the end date
                if batch_df['timestamp'].iloc[-1] >= pd.to_datetime(end_date):
                    break
            else:
                print("No more data within the specified range.")
                break  # Stop fetching if the batch is empty after filtering

            # Sleep between requests based on rate limit to avoid hitting the API rate limit
            time.sleep(exchange.rateLimit / 1000)
        except ccxt.NetworkError as e:
            print(f"Network error: {e}")
            time.sleep(5)  # Wait before retrying
        except ccxt.ExchangeError as e:
            print(f"Exchange error: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break

    # Convert all_data to a DataFrame
    df_ochlv = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df_ochlv

def load_data(file_path):
    """
    Load a CSV file from the specified file path.

    Parameters:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    pd.DataFrame: The loaded dataset.

    Raises:
    FileNotFoundError: If the file does not exist at the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    
    data = pd.read_csv(file_path)
    print(f"Dataset loaded from {file_path}.")
    
    return data

def save_data(data, file_path):
    """
    Save a DataFrame to a CSV file at the specified file path.

    Parameters:
    data (pd.DataFrame): The dataset to be saved.
    file_path (str): The path where the CSV file will be saved.

    Raises:
    ValueError: If the data is None.
    """
    if data is None:
        raise ValueError("No dataset available to save.")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the data to the specified path
    data.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

def save_model(model, file_path):
    """
    Save the given model to the specified file path.

    Parameters:
    model: The machine learning model to be saved. 
    file_path (str): The path where the model will be saved.

    Raises:
    AttributeError: If the model does not have a `save` method.
    IOError: If there is an issue saving the model to the specified file path.
    """
    model.save(file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a machine learning model from the specified file path.

    Parameters:
    file_path (str): The path to the model file to be loaded.

    Returns:
    model: The loaded machine learning model.

    Raises:
    IOError: If there is an issue loading the model from the specified file path.
    """
    model = internal_load_model(file_path)
    print(f"Model loaded from {file_path}")
    return model

def check_for_missing_data(df):
    """
    Check for missing timestamps and data points in the DataFrame.

    This function will:
    1. Convert the 'timestamp' column to datetime format if it's not already.
    2. Automatically detect the frequency of timestamps (e.g., 1 minute, hourly, etc.).
    3. Check for missing timestamps based on the detected frequency.
    4. Identify and print rows that contain missing data points (NaN values).

    Parameters:
    df (pd.DataFrame): The DataFrame to check, which should include a 'timestamp' column.

    Returns:
    None
    """
    # Ensure the timestamp column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Check the frequency of timestamps in the dataset by finding the most common difference between consecutive timestamps
    time_diffs = df['timestamp'].diff().dropna()
    most_common_diff = time_diffs.mode()[0]

    # Calculate the expected frequency based on the most common time difference
    expected_frequency = most_common_diff

    # Generate the expected range of timestamps based on the start and end of the dataset
    expected_timestamps = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq=expected_frequency)

    # Find missing timestamps
    missing_timestamps = expected_timestamps.difference(df['timestamp'])

    if not missing_timestamps.empty:
        print("Missing timestamps:")
        print(missing_timestamps)
    else:
        print("No missing timestamps.")

    # Check for rows with missing data points (NaN values)
    missing_data_rows = df[df.isna().any(axis=1)]

    if not missing_data_rows.empty:
        print("\nRows with missing data points:")
        print(missing_data_rows)
    else:
        print("\nNo missing data points.")

def distribution_plot(*data_sets, bins=30, colors=None):
    """
    Plot the distribution of multiple datasets with Kernel Density Estimate (KDE) on the same plot.

    Parameters:
    *data_sets: Multiple datasets (pd.Series or np.ndarray) to plot the distributions for.
    bins (int): The number of bins to use for the histogram. Default is 30.
    colors (list): List of colors to use for each dataset. If None, it will automatically assign colors.

    Returns:
    None
    """
    if colors is None:
        colors = sns.color_palette("Set1", len(data_sets))  # Use a color palette

    # Plot distribution with KDE for each dataset
    for i, data in enumerate(data_sets):
        sns.histplot(data, kde=True, bins=bins, color=colors[i], label=f'Data {i+1}')
    
    plt.title('Distributions with KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def profit_simulation(y_test_pred, y_test, threshold=0.008):
    """
    Simulates the profit based on predicted and actual values.
    Parameters:
        y_test_pred (list of tuples): A list of tuples where each tuple contains the predicted max and min values.
        y_test (list of tuples): A list of tuples where each tuple contains the actual max and min values.
        threshold (float, optional): The threshold value to determine if a trade should be placed. Defaults to 0.008.
    Returns:
        float: The total profit calculated based on the predictions and actual values.
    Example:
        y_test_pred = [(0.01, -0.005), (0.02, -0.01)]
        y_test = [(0.015, -0.005), (0.025, -0.015)]
        profit = profit_simulation(y_test_pred, y_test, threshold=0.01)
    """
    total_profit = 0
    for pred, actual in zip(y_test_pred, y_test):
        pred_max, pred_min = pred
        actual_max, actual_min = actual
        
        # Check if the predicted max is above the threshold to place a trade
        if pred_max >= threshold:
            # Calculate the profit based on actual max and min
            if actual_max > threshold:
                profit = actual_max
            else:
                profit = pred_min
            total_profit += profit
    
    return total_profit

def calculate_percentage_above_threshold(y_test, threshold=0.05):
    """
    Calculate the percentage of values in y_test that are above the given threshold.

    Parameters:
    y_test (numpy.ndarray): The array of test values.
    threshold (float): The threshold value to compare against.

    Returns:
    float: The percentage of values above the threshold.
    """
    percentage_above_threshold = np.mean(y_test[:, 0] > threshold) * 100
    return percentage_above_threshold