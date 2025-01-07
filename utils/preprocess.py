"""
This module contains a set of functions for preprocessing time series data related to financial markets.
The functions are designed to prepare data for machine learning models used in financial prediction tasks,
such as forecasting price movements and training models on historical OHLCV (Open, High, Low, Close, Volume) data.

Functions:
- `create_indicators`: 
    Generates a specified number of technical indicators based on their importance from a given DataFrame containing OHLCV data. 
    It uses the `TA-Lib` library to calculate various technical indicators like MFI, ADX, RSI, and MACD, among others, 
    and returns a DataFrame containing the selected indicators.

- `create_targets`: 
    Creates input windows and corresponding target values for a trading model. 
    Given a DataFrame with a 'close' column for target prices, it calculates the maximum and minimum price percentage changes 
    over a specified look-ahead period, which will serve as the model's target values.

- `normalize_X`: 
    Normalizes one or more input datasets using a specified scaler ('MinMaxScaler', 'StandardScaler', or 'RobustScaler'). 
    The 'close' column is excluded from normalization, and the normalized datasets are returned as pandas DataFrames.

- `normalize_y`: 
    Normalizes one or more target datasets using `MinMaxScaler` with a range of (-1, 1). 
    The function can return the fitted scaler for later use when normalizing other datasets.

Dependencies:
- pandas: For handling DataFrames and time series data.
- numpy: For handling numerical arrays.
- scikit-learn: For normalization functions (MinMaxScaler, StandardScaler, RobustScaler).
- TA-Lib: For calculating various technical indicators like MACD, ADX, RSI, and more.

Usage:
- `create_indicators` can be used to calculate a set of technical indicators that can be used as features for a trading model.
- `create_targets` prepares target values for training the model to predict future price changes.
- `normalize_X` and `normalize_y` are useful for scaling datasets before feeding them into machine learning models.

Example:
    # Create technical indicators for a DataFrame
    indicators_df = create_indicators(df, drop=10)
    
    # Create input-output windows and target values
    X, y = create_targets(indicators_df, window_size=50, look_ahead=10)
    
    # Normalize the input and target datasets
    X_normalized = normalize_X(indicators_df, scaler_name='MinMaxScaler')
    y_normalized = normalize_y(y, return_scaler=True)

Note: Ensure that the `TA-Lib` library is installed and properly configured before using these functions, as it is required for technical indicator calculations.
"""

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def create_indicators(dataframe, drop):
    """
    Create technical indicators for a given DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing OHLCV data.
    drop (int): The number of indicators to include based on their importance.

    Returns:
    pd.DataFrame: A DataFrame containing the calculated indicators.
    """
    # Convert 'timestamp' column to datetime
    timeframes = pd.to_datetime(dataframe['timestamp'], errors='coerce')

    
    # Dictionary to store the indicators
    indicators = {}

    # List of indicators in order of importance
    importance = ['MFI', 'HT_DCPERIOD', 'NATR', 'ATR', 'HT_DCPHASE', 'CCI', 'ULTOSC', 'volume', 'ADOSC', 'DX', 'PLUS_DM', 'PLUS_DI', 'MINUS_DI', 'SAREXT', 'ADX', 'MINUS_DM', 'RSI', 'AD', 'ADXR', 'MACD_hist', 'STOCH_d', 'CMO', 'MOM', 'TRANGE', 'OBV', 'ROC', 'WILLR', 'TRIX', 'AROON_down', 'hour', 'day', 'MACD', 'STOCH_k', 'AROONOSC', 'PPO', 'MACD_signal', 'AROON_up', 'STOCHF_k', 'ROCR', 'APO', 'STOCHF_d', 'ROCR100', 'ROCP', 'BOP', 'minute', 'HT_TRENDMODE']

    # Select the top `drop` indicators
    importance = importance[:drop]

    # Dictionary mapping indicator names to their calculation functions
    indicator_functions = {
        'SAREXT': lambda df: talib.SAREXT(df['high'], df['low']),
        'ADX': lambda df: talib.ADX(df['high'], df['low'], df['close']),
        'ADXR': lambda df: talib.ADXR(df['high'], df['low'], df['close']),
        'APO': lambda df: talib.APO(df['close']),
        'AROON_up': lambda df: talib.AROON(df['high'], df['low'])[0],
        'AROON_down': lambda df: talib.AROON(df['high'], df['low'])[1],
        'AROONOSC': lambda df: talib.AROONOSC(df['high'], df['low']),
        'BOP': lambda df: talib.BOP(df['open'], df['high'], df['low'], df['close']),
        'CCI': lambda df: talib.CCI(df['high'], df['low'], df['close']),
        'CMO': lambda df: talib.CMO(df['close']),
        'DX': lambda df: talib.DX(df['high'], df['low'], df['close']),
        'MACD': lambda df: talib.MACD(df['close'])[0],
        'MACD_signal': lambda df: talib.MACD(df['close'])[1],
        'MACD_hist': lambda df: talib.MACD(df['close'])[2],
        'MFI': lambda df: talib.MFI(df['high'], df['low'], df['close'], df['volume']),
        'MINUS_DI': lambda df: talib.MINUS_DI(df['high'], df['low'], df['close']),
        'MINUS_DM': lambda df: talib.MINUS_DM(df['high'], df['low']),
        'MOM': lambda df: talib.MOM(df['close']),
        'PLUS_DI': lambda df: talib.PLUS_DI(df['high'], df['low'], df['close']),
        'PLUS_DM': lambda df: talib.PLUS_DM(df['high'], df['low']),
        'PPO': lambda df: talib.PPO(df['close']),
        'ROC': lambda df: talib.ROC(df['close']),
        'ROCP': lambda df: talib.ROCP(df['close']),
        'ROCR': lambda df: talib.ROCR(df['close']),
        'ROCR100': lambda df: talib.ROCR100(df['close']),
        'RSI': lambda df: talib.RSI(df['close']),
        'STOCH_k': lambda df: talib.STOCH(df['high'], df['low'], df['close'])[0],
        'STOCH_d': lambda df: talib.STOCH(df['high'], df['low'], df['close'])[1],
        'STOCHF_k': lambda df: talib.STOCHF(df['high'], df['low'], df['close'])[0],
        'STOCHF_d': lambda df: talib.STOCHF(df['high'], df['low'], df['close'])[1],
        'TRIX': lambda df: talib.TRIX(df['close']),
        'ULTOSC': lambda df: talib.ULTOSC(df['high'], df['low'], df['close']),
        'WILLR': lambda df: talib.WILLR(df['high'], df['low'], df['close']),
        'AD': lambda df: talib.AD(df['high'], df['low'], df['close'], df['volume']),
        'ADOSC': lambda df: talib.ADOSC(df['high'], df['low'], df['close'], df['volume']),
        'OBV': lambda df: talib.OBV(df['close'], df['volume']),
        'HT_DCPERIOD': lambda df: talib.HT_DCPERIOD(df['close']),
        'HT_DCPHASE': lambda df: talib.HT_DCPHASE(df['close']),
        'HT_TRENDMODE': lambda df: talib.HT_TRENDMODE(df['close']),
        'ATR': lambda df: talib.ATR(df['high'], df['low'], df['close']),
        'NATR': lambda df: talib.NATR(df['high'], df['low'], df['close']),
        'TRANGE': lambda df: talib.TRANGE(df['high'], df['low'], df['close']),
        'day': lambda df: timeframes.dt.day,
        'hour': lambda df: timeframes.dt.hour,
        'minute': lambda df: timeframes.dt.minute,
        'volume': lambda df: df['volume'],
    }

    # Calculate each indicator and store in the dictionary
    for indicator in importance:
        if indicator in indicator_functions:
            indicators[indicator] = indicator_functions[indicator](dataframe)

    # Create a new DataFrame from the dictionary
    indicators_df = pd.DataFrame(indicators)
    
    # Add the 'close' column from the original dataframe
    indicators_df['close'] = dataframe['close'].iloc[indicators_df.index].values

    # Remove NaN rows
    indicators_df.dropna(inplace=True)
    
    # Reset the index
    indicators_df.reset_index(drop=True, inplace=True)
    
    return indicators_df

def create_targets(df, window_size, look_ahead, shift=1):
    """
    Create input windows and corresponding target values for a trading model.
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data with a 'close' column for target prices.
    window_size (int): The size of the window to use for creating input features.
    look_ahead (int): The number of future time steps to look ahead for calculating target values.
    shift (int, optional): The step size to move the window forward. Default is 1.
    Returns:
    tuple: A tuple containing:
        - X (np.ndarray): Array of input windows with shape (num_windows, window_size, num_features).
        - y (np.ndarray): Array of target values with shape (num_windows, 2) where each target contains 
                          the max and min percentage changes from the current price.
    """
    # Columns to use as features
    feature_columns = df.columns.drop('close')  # exclude 'close' if it's the target

    # Lists to hold windows and targets
    X_windows = []
    y_targets = []

    # Loop to create windows and look-ahead targets
    for i in range(0, len(df) - window_size - look_ahead, shift):
        # Create the input window of size `window_size`
        X_window = df.iloc[i : i + window_size][feature_columns].values
        
        # Calculate targets based on the look-ahead period
        future_prices = df['close'].iloc[i + window_size : i + window_size + look_ahead].values
        current_price = df['close'].iloc[i + window_size - 1]

        # Calculate max price and min price using the buffers
        max_price = np.max(future_prices)
        min_price = np.min(future_prices)

        # Calculate the target percentages
        target_max = (max_price - current_price) / current_price
        
        # Set the min to a negative value
        target_min = -(current_price - min_price) / current_price
                
        # Append window and target
        X_windows.append(X_window)
        y_targets.append([target_max, target_min])

        if np.isnan(X_window).any():  # detects NaN
            print('NaN in X: {X_window}')

    # Convert lists to numpy arrays for modeling
    X = np.array(X_windows)  # Shape: (num_windows, window_size, num_features)
    y = np.array(y_targets)  # Shape: (num_windows, 2) for max, min
    
    return X, y

# Define the scalers
scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
}

def normalize_X(*datasets, scaler_name):
    """
    Normalize an arbitrary number of datasets independently using the specified scaler, ignoring the 'close' column.

    Parameters:
    *datasets: Arbitrary number of pandas DataFrames representing datasets.
    scaler_name (str): The name of the scaler to use for normalization. Must be one of 'MinMaxScaler', 'StandardScaler', or 'RobustScaler'.

    Returns:
    Separated normalized datasets as pandas DataFrames.
    """
    normalized_datasets = []
    scaler = scalers[scaler_name]

    for dataset in datasets:
        # Check if the dataset is a DataFrame
        if isinstance(dataset, pd.DataFrame):
            # Separate the 'close' column
            close_column = dataset['close']
            features = dataset.drop(columns=['close'])
            
            # Normalize each column in the DataFrame except 'close'
            normalized_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
            
            # Add the 'close' column back to the normalized DataFrame
            normalized_data = pd.concat([normalized_features, close_column], axis=1)
            normalized_datasets.append(normalized_data)
        else:
            raise ValueError("All datasets must be pandas DataFrames")

    return normalized_datasets[0] if len(normalized_datasets) == 1 else tuple(normalized_datasets)


def normalize_y(*y_datasets, return_scaler=False):
    """
    Normalize an arbitrary number of y datasets independently using MinMaxScaler.
    It returns the scaled versions of the datasets and the scaler for later use if specified.
    
    Parameters:
    return_scaler (bool): Whether to return the scaler or not
    *y_datasets: Arbitrary number of datasets representing y data to be normalized
    
    Returns:
    Tuple containing:
    - A list with the scaled y datasets.
    - The fitted scaler for later use (if return_scaler=True)
    """
    
    # Initialize MinMaxScaler with range (-1, 1)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    scaled_y_datasets = []

    # Normalize the first dataset using fit_transform
    first_y_data = y_datasets[0]
    scaled_first_y_data = scaler_y.fit_transform(first_y_data)  # Fit and normalize the first dataset
    scaled_y_datasets.append(scaled_first_y_data)

    # Normalize the remaining datasets using transform
    for y_data in y_datasets[1:]:
        scaled_y_data = scaler_y.transform(y_data)  # Apply the same scaler
        scaled_y_datasets.append(scaled_y_data)

    # Return the scaled datasets, and optionally the scaler if needed
    if return_scaler:
        return (*scaled_y_datasets, scaler_y)
    return scaled_y_datasets[0] if len(scaled_y_datasets) == 1 else tuple(scaled_y_datasets)

