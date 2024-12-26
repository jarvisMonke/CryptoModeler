import ccxt
import pandas as pd
import time
import talib
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
import optuna
import numpy as np
import logging
import math

logging.basicConfig(
    level=logging.DEBUG,  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
  
)


# Fetch the data
def fetch_data(symbol, timeframe, start_date, end_date):
    
    exchange = ccxt.binanceus({
    'apiKey': 'ixiC5PZRFVXigpz97RZOju0ttiZyEBDr4gDwpALTMnF2DjgdSvtg6GrioqXOasSV',
    'secret': 'uqrpOre77kKdMpyyMmoZL8mgaGBTT8vpb5UAk53Fvcf1nkCkxSelyek0sGD2yC2q',
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
                file_path = f'data/{self.symbol.replace("/","")}_{self.timeframe}_v{self.version}.csv'
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

# Assuming you have a CSV file with columns: timestamp, open, high, low, close, volume, SMA_20, EMA_20, RSI_14, MACD, MACD_signal, MACD_hist, BB_upper, BB_middle, BB_lower, ADX_14

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

def create_targets(df, window_size, look_ahead):
    # Columns to use as features
    feature_columns = df.columns.drop('close')  # exclude 'close' if it's the target

    # Lists to hold windows and targets
    X_windows = []
    y_targets = []

    # Loop to create windows and look-ahead targets
    for i in range(len(df) - window_size - look_ahead):
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
            logging.warning(f'NaN in X: {X_window}')

    # Convert lists to numpy arrays for modeling
    X = np.array(X_windows)  # Shape: (num_windows, window_size, num_features)
    y = np.array(y_targets)  # Shape: (num_windows, 2) for max, min
    
    return(X,y)



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
    """
    Calculate a wide range of technical indicators using TA-Lib and return them as a dictionary.
    
    Parameters:
    dataframe (pd.DataFrame): A DataFrame with 'open', 'high', 'low', 'close', and 'volume' columns.
    
    Returns:
    dict: A dictionary with indicator names as keys and calculated values as values.
    """
    indicators = {}


    # Price-based Indicators
    # indicators['BBANDS_upper'], indicators['BBANDS_middle'], indicators['BBANDS_lower'] = talib.BBANDS(dataframe['close'])
    # indicators['DEMA'] = talib.DEMA(dataframe['close'])
    # indicators['EMA'] = talib.EMA(dataframe['close'])
    # indicators['HT_TRENDLINE'] = talib.HT_TRENDLINE(dataframe['close'])
    # indicators['KAMA'] = talib.KAMA(dataframe['close'])
    # indicators['MA'] = talib.MA(dataframe['close'])
    # indicators['MAMA_1'], indicators['MAMA_2'] = talib.MAMA(dataframe['close'])
    # indicators['MIDPOINT'] = talib.MIDPOINT(dataframe['close'])
    # indicators['MIDPRICE'] = talib.MIDPRICE(dataframe['high'], dataframe['low'])
    # indicators['SAR'] = talib.SAR(dataframe['high'], dataframe['low'])
    indicators['SAREXT'] = talib.SAREXT(dataframe['high'], dataframe['low'])
    # indicators['SMA'] = talib.SMA(dataframe['close'])
    # indicators['T3'] = talib.T3(dataframe['close'])
    # indicators['TEMA'] = talib.TEMA(dataframe['close'])
    # indicators['TRIMA'] = talib.TRIMA(dataframe['close'])
    # indicators['WMA'] = talib.WMA(dataframe['close'])
    
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
    
    # Price Transform
    # indicators['AVGPRICE'] = talib.AVGPRICE(dataframe['open'], dataframe['high'], dataframe['low'], dataframe['close'])
    # indicators['MEDPRICE'] = talib.MEDPRICE(dataframe['high'], dataframe['low'])
    # indicators['TYPPRICE'] = talib.TYPPRICE(dataframe['high'], dataframe['low'], dataframe['close'])
    # indicators['WCLPRICE'] = talib.WCLPRICE(dataframe['high'], dataframe['low'], dataframe['close'])
    
    # Volatility Indicators
    indicators['ATR'] = talib.ATR(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['NATR'] = talib.NATR(dataframe['high'], dataframe['low'], dataframe['close'])
    indicators['TRANGE'] = talib.TRANGE(dataframe['high'], dataframe['low'], dataframe['close'])
    
    # Time Indicators 
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], errors='coerce')
    indicators['day'] = dataframe['timestamp'].dt.day
    indicators['hour'] = dataframe['timestamp'].dt.hour
    indicators['minute'] = dataframe['timestamp'].dt.minute

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

    # Filter the tensor based on the top-ranked indices
    filtered_df = feature_df[top_ranked_indices]
    filtered_df['close'] = feature_df['close']  # Add the target column back if needed
    return filtered_df 


  # Load Dataset based on timeframe
    
class pipeline:
    def __init__(self):
        self.data = None
        self.class_name = self.__class__.__name__
        self.model = None

    def pass_hyperparams(self, timeframe, num_indicators, scaler_type, window_size, look_ahead_size, dropout_rate, prediction_tolerance_max, prediction_tolerance_min, trade_threshold):
        
        self.hyperparams = timeframe, num_indicators, scaler_type, window_size, look_ahead_size, dropout_rate, prediction_tolerance_max, prediction_tolerance_min, trade_threshold
        self.timeframe = timeframe
        self.num_indicators = num_indicators # Number of technical indicators to include
        self.scaler_type = scaler_type # the type of normalizing scaler to use
        self.window_size = window_size # the window size of the inputs
        self.look_ahead_size = look_ahead_size # the size of the look ahead period
        self.dropout_rate = dropout_rate # dropout rate in the lstm
        # For use in the stratagy simulator
        self.prediction_tolerance_max = prediction_tolerance_max 
        self.prediction_tolerance_min = prediction_tolerance_min
        self.trade_threshold = trade_threshold

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
        self.X, self.y  = create_targets(self.data, self.window_size, self.look_ahead_size)
    
    # Splits and shuffle data
    def split_data(self):

        # We want to shuffle the data to prevent the model from learning any unintended patterns from the order of the sequences
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, shuffle=True)

        # Scale the y_data
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        self.unscaled_y_train = self.y_train # create a data set to inverse the scale
        self.unscaled_y_test = self.y_test # create a data set to inverse the scale

        self.y_train = scaler_y.fit_transform(self.y_train)  # Fit and normalize
        self.y_test = scaler_y.fit_transform(self.y_test)  # Fit and normalize
        print("Data Split!")


    # Trains a model
    def train(self, epochs=50, batch_size=32):

        # Create and compile model
        height, length, width = self.X_train.shape
        model = create_model(input_shape=(length, width), dropout=self.dropout_rate)

        # Assume X_train, y_train, X_val, y_val are defined elsewhere
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Callback to check NaNs in logs during training
        class NaNChecker(tf.keras.callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                if any(np.isnan(value) for value in logs.values()):
                    print(f"NaN detected in batch {batch}")
                    self.model.stop_training = True
                    logging.debug(f"Model has NaNs. dropout: {self.dropout}")

        nan_checker = NaNChecker()

        history = model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, 
                        validation_split=0.1, 
                        verbose=1, callbacks=[early_stopping, nan_checker])
        self.model = model
    
    # save a model
    def save_Model(self, version=0):
        self.model.save(f'models/{self.class_name}_v{version}.keras')

    # load a model
    def load_Model(self, version=0):
        self.model = load_model(f'models/{self.class_name}_v{version}.keras')

    # calculate the -% profitability
    def return_profit(self):
            # get a negative profit %
        return simulate_trading(self.model, self.prediction_tolerance_max, self.prediction_tolerance_min, self.X_test, self.unscaled_y_test, self.unscaled_y_train, self.trade_threshold)
    
    # runs a full training stack
    def full_stack(self, filename):
        self.load_file(filename=filename)
        self.preprocess()
        self.target_creation()
        self.split_data()
        self.train()
        total_profit, X_test, y_test, y_train, predictions = self.return_profit()
        return total_profit
    
    

def distribution(data, bins = 30):   
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot distribution with KDE
    sns.histplot(data, kde=True, bins=bins, color='blue')
    plt.title('Distribution with KDE')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()




def create_model(input_shape, dropout):
    model = Sequential()

    # Specify the input shape using the Input() layer
    model.add(Input(shape=input_shape))

    # LSTM layer
    model.add(LSTM(128, activation='relu', kernel_initializer=HeNormal(), return_sequences=True))

    #model.add(LSTM(128, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(64, activation='relu', kernel_initializer=HeNormal(), return_sequences=False))

    #model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dropout(dropout))

    # Dense layer for output
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))  # Output layer: 2 values for predicted min and max percentage change

    optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)  # Clip gradients at 1.0
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
    none = 0

    total_profit = 0
    tot_take_profit = []
    tot_stop_loss = []
    print(f'Predictions:{predictions_rescaled[:10]}')
    print(f'Y-Tests:{y_test[:10]}')

    # Go through all the predictions 
    for frame in range(len(predictions_rescaled)):
        # If the trade_threshold is reached place a trade 
        if predictions_rescaled[frame][0] >= trade_threshold:
            
            # Calculate the take profits
            take_profit = predictions_rescaled[frame][0] * prediction_tolerance_max
            tot_take_profit.append(take_profit)

            # Calcuate the stop losses
            stop_loss = predictions_rescaled[frame][1] * prediction_tolerance_min
            tot_stop_loss.append(stop_loss)

            # If the trade was profitable return the profit
            if take_profit >= y_test[frame][0]:
                profit = y_test[frame][0]
                wins += 1
            # Unprofitable trades return the stoploss
            else:
                # Can have a >1 stop loss
                profit = stop_loss - int(stop_loss)
                loses += 1
            # taker fee + maker fee
            fee = .006 + .004*(1+profit)
            total_profit = total_profit + profit - fee
        else:
            tot_take_profit.append(0)
            tot_stop_loss.append(0)
            none += 1
    
    print(f'TP:{tot_take_profit[:10]}')
    print(f'SL:{tot_stop_loss[:10]}')

    # Return negative profit as something to minamize
    print(f'Tot Profit: {total_profit}, Wins: {wins}, Loses: {loses}, No-trade: {none}')
    return -total_profit, X_test, y_test, y_train, predictions_rescaled