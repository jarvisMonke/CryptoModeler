import ccxt
import pandas as pd
import time
import talib
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import configparser

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import HeNormal

from optuna.integration import KerasPruningCallback  
from optuna.exceptions import TrialPruned

# Set up the path to the config file
config_path = Path(__file__).resolve().parents[1] / 'config' / 'config.ini'

# Ensure the config file exists
if not config_path.exists():
    raise FileNotFoundError(f"Config file not found at {config_path}.")

# Create a ConfigParser instance and read the config file
config = configparser.ConfigParser()
config.read(config_path)

# Fetch the data
def fetch_data(symbol, timeframe, start_date, end_date):
    
    exchange = ccxt.binanceus({
        'apiKey': config['BINANCEUS']['API_KEY'],
        'secret': config['BINANCEUS']['SECRET']
    })
    limit = 500
    start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)

    # Fetch data in batches 
    all_data = []
    while start_timestamp < end_timestamp:
        ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=start_timestamp, limit=limit)
        
        if not ohlcv_data:
            break

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

        time.sleep(exchange.rateLimit / 1000)  # sleep between requests based on rate limit

    # Convert all_data to a DataFrame
    df_ochlv = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df_ochlv



# CREATE/SAVE/LOAD a dataset returns a dataframe
class CryptoDataset:
    def __init__(self, symbol, timeframe, start_date, end_date, version=1):

        self.symbol = symbol # Symbol: DOGE/USD
        self.timeframe = timeframe # Timeframe: 5m
        self.start_date = start_date # Start/end date:  '2023-01-01 00:00:00'
        self.end_date = end_date
        self.version = version # Version: a name or number used for identifying similar datasets
        self.data = None
        self.default_file_path = None  # No default file path initially

    # Retrives data from binance, values set in __init__
    def fetch(self):
        
        self.data = fetch_data(symbol=self.symbol, timeframe=self.timeframe, start_date=self.start_date, end_date=self.end_date)
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("The data source function must return a pandas DataFrame.")

     # Loads data from a CSV to self.data
    def load(self, file_path=None, class_name=None, data_type=None):
        
        # If no custom file path is provided, use the default or generate the default path
        if file_path is None:
            if self.default_file_path is None:
                file_path = f'data/{self.symbol.replace("/","")}_{class_name}_{data_type}.csv'
            else:
                file_path = self.default_file_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}.")
        
        self.data = pd.read_csv(file_path)
        print(f"Dataset loaded from {file_path}.")
        
        # Update default path to the one used (if it's a custom one)
        self.default_file_path = file_path

    # Saves whatever is currently inside self.data to a CSV
    def save(self, file_path=None):
        
        # If no custom file path is provided, use the default or generate the default path
        if file_path is None:
            if self.default_file_path is None:
                # TODO save to the proper directory
                file_path = f'./data/{self.symbol.replace("/","")}_{self.timeframe}_v{self.version}.csv'
            else:
                file_path = self.default_file_path

        if self.data is None:
            raise ValueError("No dataset available to save.")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the data to the specified path
        self.data.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}.")
        
        # Update default path to the one used (if it's a custom one)
        self.default_file_path = file_path


# CHECK FOR MISSING DATA
def check_for_missing_data(df):
     # Ensure the timestamp column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Check for missing timestamps
    # We assume 1-minute interval between each timestamp, adjust if needed
    expected_timestamps = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='min')

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


# DATA NORMILIZATION


# Initialize the chosen scaler
scalers = {
    'MinMaxScaler': MinMaxScaler(),
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
}

# Function to normalize the DataFrame based on the specified scaler
def normalize_data(df, scaler_name):
    # Drop timestamps if needed
    # df = df.drop(columns=['timestamp'])
    # Select the appropriate scaler
    scaler = scalers[scaler_name]

    # Exclude the target column ('close' or others you don't want to normalize)
    features = df.drop(columns=['close']).copy()  # Adjust if other columns should be excluded

    # Apply the chosen scaler and convert the scaled data to float32
    df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, dtype='float32')

    # Include the target column back
    df_scaled['close'] = df['close']  # Add the target column back if needed

    return df_scaled


# TARGET CREATION
def create_targets(df, window_size, look_ahead, shift=1):
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


# Calculate the number of windows with a max percent above n
def profitable_percent(y, percent):
    count = 0
    y = [pair[0] for pair in y]
    list_of_windows = [] 
    for i in y:
        if i >= percent:
            count += 1
            list_of_windows.append(i)
        else: 
            pass
    average = sum(list_of_windows)/len(list_of_windows)
    return count, "{:.4f}".format(average*100)

# CREATE FEATURES

def create_features(dataframe):
  
    indicators = {}

    indicators['SAREXT'] = talib.SAREXT(dataframe['high'], dataframe['low'])
    # Momentum Indicators
    indicators['ADX'] = talib.ADX(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['ADXR'] = talib.ADXR(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['APO'] = talib.APO(dataframe['close'])
    indicators['AROON_up'], indicators['AROON_down'] = talib.AROON(dataframe['high'], dataframe['low'])
    indicators['AROONOSC'] = talib.AROONOSC(dataframe['high'], dataframe['low'])
    indicators['BOP'] = talib.BOP(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['CCI'] = talib.CCI(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['CMO'] = talib.CMO(dataframe['close'])
    indicators['DX'] = talib.DX(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['MACD'], indicators['MACD_signal'], indicators['MACD_hist'] = talib.MACD(dataframe['close'])
    indicators['MFI'] = talib.MFI(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
    indicators['MINUS_DI'] = talib.MINUS_DI(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['MINUS_DM'] = talib.MINUS_DM(dataframe['high'], dataframe['low'])
    indicators['MOM'] = talib.MOM(dataframe['close'])
    indicators['PLUS_DI'] = talib.PLUS_DI(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['PLUS_DM'] = talib.PLUS_DM(dataframe['high'], dataframe['low'])
    indicators['PPO'] = talib.PPO(dataframe['close'])
    indicators['ROC'] = talib.ROC(dataframe['close'])
    indicators['ROCP'] = talib.ROCP(dataframe['close'])
    indicators['ROCR'] = talib.ROCR(dataframe['close'])
    indicators['ROCR100'] = talib.ROCR100(dataframe['close'])
    indicators['RSI'] = talib.RSI(dataframe['close'])
    indicators['STOCH_k'], indicators['STOCH_d'] = talib.STOCH(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['STOCHF_k'], indicators['STOCHF_d'] = talib.STOCHF(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['TRIX'] = talib.TRIX(dataframe['close'])
    indicators['ULTOSC'] = talib.ULTOSC(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['WILLR'] = talib.WILLR(dataframe['high'], dataframe['low'], dataframe['close'])
    
    # Volume Indicators
    indicators['AD'] = talib.AD(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
    indicators['ADOSC'] = talib.ADOSC(dataframe['high'], dataframe['low'], dataframe['close'], dataframe['volume'])
    indicators['OBV'] = talib.OBV(dataframe['close'], dataframe['volume'])
    
    # Cycle Indicators
    indicators['HT_DCPERIOD'] = talib.HT_DCPERIOD(dataframe['close'])
    indicators['HT_DCPHASE'] = talib.HT_DCPHASE(dataframe['close'])
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(dataframe['close'])
    
    # Volatility Indicators
    indicators['ATR'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['NATR'] = talib.NATR(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['TRANGE'] = talib.TRANGE(dataframe['high'], dataframe['low'], dataframe['close'])
    
    # Time Indicators 
    # Convert the timestamp column to datetime 
    timeframes = pd.to_datetime(dataframe['timestamp'], errors='coerce')
    
    indicators['day'] = timeframes.dt.day
    indicators['hour'] = timeframes.dt.hour
    indicators['minute'] = timeframes.dt.minute

    # Create a new DataFrame from the dictionary
    new_cols = pd.DataFrame(indicators)

    # Concatenate the new DataFrame with the original DataFrame
    indicators_df = pd.concat([dataframe, new_cols], axis=1)
    indicators_df.dropna(inplace=True)  # Remove NaN rows
    
    # the NaN bug fix (maybe)
    indicators_df.reset_index(drop=True, inplace=True)  # Reset the index
       
    indicators_df = indicators_df.drop(columns=['timestamp'])

    indicators_df = indicators_df.drop(columns=['high'])
    indicators_df = indicators_df.drop(columns=['low'])
    indicators_df = indicators_df.drop(columns=['open'])


    #indicators_df = pd.DataFrame(indicators, index=dataframe.index)
    return indicators_df


# CUT FEATURES based on the list
def feature_cut(feature_df, indicator_count):
   
    importance_list = [
    ('MFI', 0.0846),
    ('HT_DCPERIOD', 0.0738),
    ('NATR', 0.0611),
    ('ATR', 0.0578),
    ('HT_DCPHASE', 0.0466),
    ('CCI', 0.0338),
    ('ULTOSC', 0.0312),
    ('volume', 0.0307),
    ('ADOSC', 0.0303),
    ('DX', 0.0291),
    ('PLUS_DM', 0.0277),
    ('PLUS_DI', 0.0258),
    ('MINUS_DI', 0.0255),
    ('SAREXT', 0.0238),
    ('ADX', 0.0231),
    ('MINUS_DM', 0.0231),
    ('RSI', 0.0229),
    ('AD', 0.0224),
    ('ADXR', 0.0194),
    ('MACD_hist', 0.0164),
    ('STOCH_d', 0.0158),
    ('CMO', 0.0158),
    ('MOM', 0.0152),
    ('TRANGE', 0.0148),
    ('OBV', 0.0136),
    ('ROC', 0.0112),
    ('WILLR', 0.0109),
    ('TRIX', 0.0109),
    ('AROON_down', 0.0109),
    ('hour', 0.0106),
    ('day', 0.0103),
    ('MACD', 0.0102),
    ('STOCH_k', 0.0101),
    ('AROONOSC', 0.0096),
    ('PPO', 0.0085),
    ('MACD_signal', 0.0080),
    ('AROON_up', 0.0077),
    ('STOCHF_k', 0.0075),
    ('ROCR', 0.0074),
    ('APO', 0.0073),
    ('STOCHF_d', 0.0067),
    ('ROCR100', 0.0066),
    ('ROCP', 0.0055),
    ('BOP', 0.0053),
    ('minute', 0.0040),
    ('HT_TRENDMODE', 0.0016)
]
    # Rank the importance scores
    sorted_feature_names = [feature for feature, _ in sorted(importance_list, key=lambda x: x[1], reverse=True)]

    # Select the top-ranked rows
    top_ranked_indices = sorted_feature_names[:indicator_count]

    # Filter the tensor based on the top-ranked indices and ensure a copy is made
    filtered_df = feature_df[top_ranked_indices].copy()

    # Add the target column back, now that filtered_df is a separate object
    filtered_df.loc[:, 'close'] = feature_df['close']

    return filtered_df 


  # Load Dataset based on timeframe
    
def split_train_val_test(X, y, train_size, val_size, test_size):
    
    num_windows = len(X)
    
    # Calculate the split indices
    train_end = int(num_windows * train_size)
    val_end = train_end + int(num_windows * val_size)
    
    # Split the windows
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_train = y[:train_end]
    y_val = y[train_end:val_end]
    y_test = y[val_end:]
    
    # Shuffle the data
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate a random permutation of indices
    train_indices = np.random.permutation(len(X_train))
    val_indices = np.random.permutation(len(X_val))
    test_indices = np.random.permutation(len(X_test))

    # Set the data to the indices
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    X_val = X_val[val_indices]
    y_val = y_val[val_indices]

    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test


class pipeline:
    def __init__(self):
        self.data = None
        self.class_name = self.__class__.__name__
        self.model = None

    def pass_hyperparams(self, timeframe, num_indicators, scaler_type, window_size, look_ahead_size, params_grid, tuning=True, prediction_tolerance_max=1, prediction_tolerance_min=1, trade_threshold=.01):
        
        self.hyperparams = timeframe, num_indicators, scaler_type, window_size, look_ahead_size, params_grid, prediction_tolerance_max, prediction_tolerance_min, trade_threshold
        
        self.timeframe = timeframe
        self.num_indicators = num_indicators # Number of technical indicators to include
        self.scaler_type = scaler_type # the type of normalizing scaler to use
        self.window_size = window_size # the window size of the inputs
        self.look_ahead_size = look_ahead_size # the size of the look ahead period
        self.params_grid = params_grid # dropout rate in the lstm
        # For use in the stratagy simulator
        self.prediction_tolerance_max = prediction_tolerance_max 
        self.prediction_tolerance_min = prediction_tolerance_min
        self.trade_threshold = trade_threshold
        self.tuning = tuning

    # Loads a file based on a filename into .data
    def load_file(self, datatype=None, filename=None):
        dataset = CryptoDataset('symbol', 'timeframe', 'start_time', 'end_time', 'o')
        if not filename:
            filename = f'data/{self.class_name}_{datatype}.csv'
        dataset.load(filename)
        self.data = dataset.data
        print(f"Loaded into {self.class_name}.data")

    # Fetchs data from binance into .data, dosn't save
    def fetch_data(self, symbol, start_data, end_date):
        dataset = CryptoDataset(symbol, self.timeframe, start_data, end_date, 'o')
        dataset.fetch()
        self.data = dataset.data
        print(f"Fetched {symbol} data into {self.class_name}.data")

    # saves what ever is currently in .data
    def save_data(self, datatype=None, filename=None):
        dataset = CryptoDataset(self.symbol, self.timeframe, 'start_data', 'end_date', 'o')
        dataset.data = self.data
        if not filename:
            filename = f'data/{self.class_name}_{datatype}.csv'
        dataset.save()

    # Creates features and cuts features
    def preprocess(self): 
        model_data = self.data

        # Create features to num_indicators
        model_data = feature_cut(create_features(model_data), self.num_indicators)

        # Normalize data based on scaler_type
        model_data = normalize_data(model_data, self.scaler_type)

        self.data = model_data
        print('Data Preprocessed!')
    
    # Creates targets 
    def target_creation(self):
         # Create Targets based on window_size and look_ahead_size
        self.X, self.y  = create_targets(self.data, self.window_size, self.look_ahead_size, shift=20)
    
    # Splits and shuffle data
    def split_data(self):

        split_index = int(len(self.X) * 0.9)
        self.X_model, self.X_test = self.X[:split_index], self.X[split_index:]
        self.y_model, self.y_test = self.y[:split_index], self.y[split_index:]

        print("Data Split into model and test Sets")
    
    def normalize_y_data(self):
         # Scale the y_data
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        self.unscaled_y_train = self.y_model # create a data set to inverse the scale
        self.unscaled_y_test = self.y_test # create a data set to inverse the scale

        self.y_model = scaler_y.fit_transform(self.y_model)  # Fit and normalize
        self.y_test = scaler_y.fit_transform(self.y_test)  # Fit and normalize
    
    def train(self, trial, epochs=50):
        """
        Train the model using the Optuna pruning callback, early stopping, and NaN checking.
        """
        # Callback to check NaNs in logs during training
        class NaNChecker(tf.keras.callbacks.Callback):
            def __init__(self, trial):
                super().__init__()
                self.trial = trial

            def on_batch_end(self, batch, logs=None):
                if logs is not None and any(np.isnan(value) for value in logs.values()):
                    print("NaN detected, pruning trial.")
                    raise optuna.exceptions.TrialPruned()
        
        # Initialize NaNChecker
        nan_checker = NaNChecker(trial)

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Create and compile the model
        height, length, width = self.X_model.shape
        model = create_model(input_shape=(length, width), params_grid=self.params_grid)

        
        # Parameters for rolling window
        train_size = 3000  # Initial training size
        val_size = 500     # Validation size
        step_size = 200    # How much to move the window for each epoch

        epoch_counter = 0  # Track total epochs across windows
        # Loop through epochs and update train and validation data
        for epoch in range(epochs):
            # For each epoch, update the training and validation data using the rolling window
            train_start = epoch * step_size
            train_end = train_start + train_size
            
            val_start = train_end
            val_end = val_start + val_size
            
            # Make sure we don't go out of bounds for the last epoch
            if val_end > len(self.X_model):
                break
            
              # Get the new training and validation data slices for the current epoch
            X_train_epoch = self.X_model[train_start:train_end]
            y_train_epoch = self.y_model[train_start:train_end]
            X_val_epoch = self.X_model[val_start:val_end]
            y_val_epoch = self.y_model[val_start:val_end]
            
            # Use Optuna's KerasPruningCallback to prune the trial if necessary
            pruning_callback = KerasPruningCallback(trial, 'val_loss')  # Monitor 'val_loss' for pruning

            # Now train the same model on the new data for this epoch (do not reinitialize model)
            history = model.fit(
                X_train_epoch, 
                y_train_epoch, 
                epochs=1,  # Train for 1 epoch at a time in this loop
                batch_size=self.params_grid["batch_size"],
                validation_data=(X_val_epoch, y_val_epoch),  # Provide the new validation data
                verbose=1, 
                callbacks=[early_stopping, nan_checker, pruning_callback]   
            )
                       
            if self.tuning:
                # Manually increment the Optuna pruning step
                trial.report(history.history['val_loss'][-1], step=epoch_counter)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                epoch_counter += 1

                # Stop if early stopping is triggered
                if early_stopping.stopped_epoch > 0:
                    print("Early stopping triggered.")
                    break


        self.model = model
        return model


    # save a model
    def save_Model(self, version=0):
        self.model.save(f'models/{self.class_name}_v{version}.keras')

    # load a model
    def load_Model(self, version=0):
        self.model = load_model(f'models/{self.class_name}_v{version}.keras')

    # calculate the % profitability
    def return_profit(self):
            # get a profit %
        return simulate_trading(self.model, self.prediction_tolerance_max, self.prediction_tolerance_min, self.X_test, self.unscaled_y_test, self.unscaled_y_train, self.trade_threshold)
    
    # runs a full training stack
    def full_stack(self, trial, filename):
        self.load_file(filename=filename)
        self.preprocess()
        self.target_creation()
        self.split_data()
        self.normalize_y_data()
        #total_profit, total_trades, X_test, y_test, y_train, predictions = self.return_profit()
        return self.train(trial=trial)

    

def create_model(input_shape, params_grid):
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # LSTM layers with dynamic units and dropout
    model.add(LSTM(params_grid["lstm_units_1"], activation=params_grid["activation"], kernel_initializer=HeNormal(), return_sequences=True))
    model.add(Dropout(params_grid["dropout"]))
    model.add(LSTM(params_grid["lstm_units_2"], activation=params_grid["activation"], kernel_initializer=HeNormal(), return_sequences=False))
    model.add(Dropout(params_grid["dropout"]))

    # Dense layers
    model.add(Dense(params_grid["dense_units"], activation=params_grid["activation"]))
    model.add(Dense(2))  # Output layer: 2 values for predicted min and max percentage change

    # Optimizer selection
    if params_grid["optimizer"] == 'adam':
        optimizer = Adam(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])
    else:
        optimizer = RMSprop(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model


# trade_threshold: the predicted % above current price to place a trade at
# prediction_tolerance_max: The percent of the predicted % to place the tp
# prediction_tolerance_min:  The percent of the predicted % to place the sl
# Trading Fee is small (0.1%)

# Function tests a model to check its profitability
def simulate_trading(model, prediction_tolerance_max, prediction_tolerance_min, X_test, y_test, y_train, 
trade_threshold):
    
    # Create prediictions based on X_test
    predictions = model.predict(X_test)
   
    scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Scale to [-1, 1]
    scaler_y.fit(y_train)

    predictions_rescaled = scaler_y.inverse_transform(predictions)

    wins = 0
    loses = 0
    no_hit = 0
    none = 0
    total_trades = 0

    total_profit = 0
    tot_take_profit = []
    tot_stop_loss = []
    print(f'Predictions:{predictions_rescaled[:10]}')
    print(f'Y-Tests:{y_test[:10]}')

    # Go through all the predictions 
    for frame in range(len(predictions_rescaled)):
        # If the trade_threshold is reached place a trade 
        if predictions_rescaled[frame][0] >= trade_threshold:
            total_trades += 1
            # Calculate the take profits
            take_profit = predictions_rescaled[frame][0] * prediction_tolerance_max
            tot_take_profit.append(take_profit)

            # Calcuate the stop losses
            stop_loss = predictions_rescaled[frame][1] * prediction_tolerance_min
            tot_stop_loss.append(stop_loss)

            if y_test[frame][0] >= take_profit:  # Profit target reached
                profit = take_profit
                wins += 1
            elif y_test[frame][1] <= stop_loss:  # Stop loss triggered
                profit = stop_loss
                loses += 1

                # Penalize Postive Stop Losses on bad models
                if stop_loss > 0:
                    profit *= -1
                    
            else:  # Trade held till the end of the frame
                profit = 0  # Actual outcome
                no_hit += 1

            # taker fee + maker fee
            fee = .006 + (.004*(1+profit))
            total_profit = total_profit + profit - fee
        else:
            tot_take_profit.append(0)
            tot_stop_loss.append(0)
            none += 1
    
    print(f'TP:{tot_take_profit[:10]}')
    print(f'SL:{tot_stop_loss[:10]}')

    # Return negative profit as something to minamize
    print(f'Tot Profit: {total_profit}, Wins: {wins}, Loses: {loses}, No-Hits: {no_hit} No-trade: {none}')
    return total_profit, total_trades, X_test, y_test, y_train, predictions_rescaled

def distribution(data, bins = 30):   

    # Plot distribution with KDE
    sns.histplot(data, kde=True, bins=bins, color='blue')
    plt.title('Distribution with KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()