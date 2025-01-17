"""
This module provides functionality for building, training, and tuning an LSTM-based machine learning model for predicting
minimum and maximum percentage changes from historical data. It integrates with Optuna for hyperparameter optimization,
supports early stopping, includes dynamic data slicing for training and validation, and offers custom callbacks during training.

Functions:
create_model(input_shape, params_grid) -> keras.models.Sequential
    - Builds and compiles a Sequential LSTM model for time series forecasting based on the provided input shape and hyperparameters.
    
data_slicer(X, y, epoch, train_size, val_size, step_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    - Slices the dataset into rolling training and validation sets for each epoch.
    
optuna_pruning_and_callbacks(trial, model, X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch, params_grid, epoch_counter, early_stopping) -> History
    - Manages Optuna pruning and applies callbacks for model training, including validation loss monitoring and early stopping.
    
train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=False, trial=None) -> keras.models.Sequential
    - Trains an LSTM model with dynamic data slicing for each epoch, optionally performing hyperparameter tuning with Optuna.
    - Includes early stopping to prevent overfitting.
    
load_model_test_split(filenames, params, model_data_ratio=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]
    - Loads data, creates technical indicators, and splits it into training and testing sets based on a specified ratio.
    
test_data(df, params, model_name=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    - Processes and normalizes the test data, optionally saving the normalized data and real targets to disk.

custom_model(df, params, model_name=None, tuning=False, trial=None) -> keras.models.Sequential
    - Trains a custom LSTM model using the provided dataframe and parameters, with options for Optuna tuning and model saving.

Notes:
- This module facilitates the building and training of an LSTM-based model for time series forecasting tasks.
- The `train` function supports dynamic data slicing to process the training data in smaller rolling windows, improving model robustness.
- Optuna integration allows for hyperparameter tuning to optimize the model’s performance.
- Early stopping is employed to avoid overfitting, halting training when the validation loss fails to improve for a certain number of epochs.
- The module includes preprocessing functions for handling both features and targets, enabling better model generalization.

Example usage:

# Train a model without Optuna tuning
model = train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200)

# Train a model with Optuna tuning
model = train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=True, trial=trial)
"""


# Import necessary libraries
import os
import numpy as np
import joblib
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from utils.preprocess import normalize_X, normalize_y, create_targets, create_indicators
from utils.helpers import load_data

def create_model(input_shape, params_grid):
    """
    Creates and compiles an LSTM-based neural network model for predicting 
    minimum and maximum percentage changes.

    Parameters:
    input_shape (tuple): Shape of the input data (excluding batch size).
    params_grid (dict): Dictionary containing hyperparameters for the model.
        - lstm_units_1 (int): Number of units in the first LSTM layer.
        - lstm_units_2 (int): Number of units in the second LSTM layer.
        - activation (str): Activation function to use in the LSTM and Dense layers.
        - dropout (float): Dropout rate for the Dropout layers.
        - dense_units (int): Number of units in the Dense layer.
        - optimizer (str): Optimizer to use ('adam' or 'rmsprop').
        - learning_rate (float): Learning rate for the optimizer.
        - gradient_clipping (float): Gradient clipping value for the optimizer.

    Returns:
    keras.models.Sequential: Compiled LSTM-based neural network model.
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # First LSTM layer with dropout
    x = LSTM(params_grid["lstm_units_1"], activation=params_grid["activation"], 
             kernel_initializer=HeNormal(), return_sequences=True)(inputs)
    x = Dropout(params_grid["dropout"])(x)

    # Second LSTM layer with dropout
    x = LSTM(params_grid["lstm_units_2"], activation=params_grid["activation"], 
             kernel_initializer=HeNormal(), return_sequences=False)(x)
    x = Dropout(params_grid["dropout"])(x)

    # Dense layer
    x = Dense(params_grid["dense_units"], activation=params_grid["activation"])(x)

    # Output heads
    output_min = Dense(1, name='max_change')(x)  # Head for min percentage change
    output_max = Dense(1, name='min_change')(x)  # Head for max percentage change

    # Create the model
    model = Model(inputs=inputs, outputs=[output_max, output_min])

    # Select optimizer and compile the model
    if params_grid["optimizer"] == 'adam':
        optimizer = Adam(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])
    else:
        optimizer = RMSprop(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])

    model.compile(
    optimizer=optimizer,
    loss=['mse', 'mse'],  # Assuming you want to use mean squared error for both outputs
    metrics=[['mae'], ['mae']]  # Provide a list of metrics for each output
)
    return model

def train(X, y, params, tuning=False, trial=None):
    """
    Train the model with early stopping, data slicing, and optional Optuna hyperparameter tuning.

    Parameters:
    - X (pd.DataFrame or np.array): Input feature data for training and validation.
    - y (pd.Series or np.array): Target labels for training and validation.
    - params (dict): Dictionary containing the model parameters, such as 'window_size', 'num_indicators', 'epochs', 
      'train_size', 'val_size', 'step_size', and other hyperparameters for model creation.
    - tuning (bool, optional): Flag to indicate if Optuna hyperparameter tuning should be applied. Default is False.
    - trial (optuna.trial.Trial, optional): The Optuna trial object to track the tuning process if `tuning=True`.

    Returns:
    - model: The trained model after completing the specified number of epochs or early stopping.

    Notes:
    - The function supports both training with and without Optuna-based hyperparameter tuning.
    - Uses early stopping to prevent overfitting by monitoring validation loss and restoring the best model weights.
    - The `data_slicer` function slices the dataset into training and validation sets for each epoch using a rolling window.
    - The model is trained for one epoch at a time, updating the training and validation data dynamically.
    - If early stopping is triggered, training is stopped prematurely.

    Example:
    ```python
    model = train(X, y, params, tuning=True, trial=trial)
    ```
    """
    from utils.tuning import optuna_pruning_and_callbacks

    # model params
    model_keys = [
        'lstm_units_1', 'lstm_units_2', 'dropout', 'learning_rate', 'batch_size',
        'dense_units', 'sequence_length', 'gradient_clipping', 'optimizer', 'activation'
    ]
    # Create the new dictionary using dictionary comprehension
    params_grid = {key: params[key] for key in model_keys}
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Create and compile the model
    length, width = params['window_size'], params['num_indicators']
    model = create_model(input_shape=(length, width), params_grid=params_grid)

    # Parameters for rolling window
    epoch_counter = 0  # Track total epochs across windows

    # Loop through epochs and update train and validation data
    for epoch in range(params['epochs']):
        # Get the new training and validation data slices for the current epoch
        X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch = data_slicer(X, y, epoch, params['train_size'], params['val_size'], params['step_size'])

        if X_train_epoch is None:
            print("Reached end of data, stopping training.")
            break
        
        # Split the tuples into two arrays
        y_max_train_epoch = np.array([y[0] for y in y_train_epoch])
        y_min_train_epoch = np.array([y[1] for y in y_train_epoch])

        y_max_val_epoch = np.array([y[0] for y in y_val_epoch])
        y_min_val_epoch = np.array([y[1] for y in y_val_epoch])

        if tuning:
            # Call Optuna-specific logic if tuning is enabled
            history = optuna_pruning_and_callbacks(trial, model, X_train_epoch, [y_max_train_epoch, y_min_train_epoch], X_val_epoch, [y_max_val_epoch, y_min_val_epoch], params_grid, epoch_counter, early_stopping)
        else:

            # Train the model without Optuna
            history = model.fit(
                X_train_epoch,
                [y_max_train_epoch, y_min_train_epoch],
                epochs=1,  # Train for 1 epoch at a time in this loop
                batch_size=params_grid["batch_size"],
                validation_data=(X_val_epoch, [y_max_val_epoch, y_min_val_epoch]),  # Provide the new validation data
                verbose=1,
                callbacks=[early_stopping]
            )
        
        epoch_counter += 1

        # Stop if early stopping is triggered
        if early_stopping.stopped_epoch > 0:
            print("Early stopping triggered.")
            break

    return model

def load_model_test_split(filenames, params, model_data_ratio=0.8):
    """
    Loads data, creates indicators, and splits the data into model and test sets.

    Parameters:
        filenames (dict): A dictionary containing file paths for different timeframes.
        params (dict): A dictionary containing parameters for loading data and creating indicators.
            - 'timeframe' (str): The key to access the appropriate file path from filenames.
            - 'num_indicators' (int): The number of indicators to create.
        model_data_ratio (float, optional): The ratio of data to be used for the model. Defaults to 0.8.
    Returns:
        tuple: A tuple containing two DataFrames:
            - model_df (DataFrame): The DataFrame to be used for the model.
            - test_df (DataFrame): The DataFrame to be used for testing.
    """
    # Load data
    df = load_data(filenames[params['timeframe']])
    # Create indicators
    indicated_df = create_indicators(df, params['num_indicators'])
    
    # Split data
    model_df = indicated_df[:int(len(indicated_df)*model_data_ratio)]
    test_df = indicated_df[int(len(indicated_df)*model_data_ratio):]
    
    return model_df, test_df

def test_data(df, params, model_name=None):
    """
    Processes and normalizes test data, then saves the processed data to disk.

    Parameters:
        df (pandas.DataFrame): The input dataframe containing the test data.
        params (dict): A dictionary containing the parameters for data processing, including:
            - 'scaler_type' (str): The type of scaler to use for normalization.
            - 'window_size' (int): The window size for creating targets.
            - 'look_ahead_size' (int): The look-ahead size for creating targets.
        model_name (str, optional): The name of the model for saving the test data. Defaults to None.
    Returns:
    tuple: A tuple containing:
        - norm_X_test (numpy.ndarray): The normalized test features.
        - norm_y_test (numpy.ndarray): The normalized test targets.
        - y_test (numpy.ndarray): The unscaled test targets.
    """

    # Testing data
    test_df = normalize_X(df, scaler_name=params['scaler_type'])

    # Testing data special treatment
    norm_X_test, y_test = create_targets(test_df, params['window_size'], params['look_ahead_size'], params['window_shift'])
    norm_y_test = normalize_y(y_test)
    
    if model_name is not None:
        os.makedirs(os.path.dirname(f'../models/{model_name}/'), exist_ok=True)

        joblib.dump(norm_X_test, f'../models/{model_name}/X_normTest.pkl')
        joblib.dump(norm_y_test, f'../models/{model_name}/y_normTest.pkl')
        joblib.dump(y_test, f'../models/{model_name}/y_realTest.pkl')
        print('Test data dumped')
    
    return norm_X_test, norm_y_test, y_test

def custom_model(df, params, model_name=None, tuning=False, trial=None):
    """
    Trains a custom model using the provided dataframe and parameters, and optionally saves the scaler for future use.

    Parameters:
    - df (pd.DataFrame): The input dataframe containing the data.
    - params (dict): A dictionary of parameters for the model, including:
        - 'scaler_type' (str): The type of scaler to use for normalization (e.g., 'StandardScaler', 'MinMaxScaler').
        - 'window_size' (int): The size of the window for creating features.
        - 'look_ahead_size' (int): The size of the look-ahead window for creating targets.
    - model_name (str, optional): The name of the model for saving the scaler. Defaults to None. If provided, saves the scalers to the specified path.
    - tuning (bool, optional): Whether the model is being tuned using Optuna. Defaults to False.
    - trial (optuna.trial.Trial, optional): The Optuna trial object for hyperparameter tuning, if `tuning=True`. Defaults to None.

    Returns:
    - model: The trained model.
    - scaler_y (optional): The scaler used to normalize the target, returned only if `tuning=True`.

    Notes:
    - The function performs preprocessing on the data using the specified scaler and creates features/targets with the given window sizes.
    - If `model_name` is provided, the scaler objects for features (`scaler_X`) and target (`scaler_y`) are saved to disk.
    - Supports hyperparameter tuning with Optuna if `tuning=True`, and returns the trained model and target scaler.
    - The model is trained using the `train` function, which incorporates early stopping and optional Optuna-based tuning.

    Example:
    ```python
    model, scaler_y = custom_model(df, params, model_name="my_model", tuning=True, trial=trial)
    ```
    """

    # Load data and create indicators
    model_df, scaler_X = normalize_X(df, scaler_name=params['scaler_type'], return_scaler=True)
    X_train, y_train = create_targets(model_df, params['window_size'], params['look_ahead_size'], params['look_ahead_size'])
    
    y_train, scaler_y = normalize_y(y_train, return_scaler=True)
    
    if model_name is not None:
        os.makedirs(os.path.dirname(f'../models/{model_name}/'), exist_ok=True)
        
        joblib.dump(scaler_X, f'../models/{model_name}/scaler_X.pkl')
        joblib.dump(scaler_y, f'../models/{model_name}/scaler_y.pkl')
        print('Scalers dumped')
    
    if params['shuffle']:
        # Set a random seed for reproducibility
        np.random.seed(42)

        # Assuming X has shape (samples, window_size, features) and y has shape (samples,)
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        X_train = X_train[indices]
        y_train = y_train[indices]

    # Train
    model = train(X_train, y_train, params, tuning=tuning, trial=trial) 
    
    if tuning:
        return model, scaler_y
    else:
        return model
    

def data_slicer(X, y, epoch, train_size, val_size, step_size):
    """
    Slice the data for the current epoch using a rolling window, create training and validation sets, and return them.

    Parameters:
    - X (np.ndarray): Input feature data.
    - y (np.ndarray): Input target data.
    - epoch (int): The current epoch number, used to calculate the starting index for slicing.
    - train_size (int): Number of samples to include in the training set.
    - val_size (int): Number of samples to include in the validation set.
    - step_size (int): The size of the shift for each rolling window (i.e., how many steps to move each time).

    Returns:
    - X_train_epoch (np.ndarray): Feature data slice for the training set in the current epoch.
    - y_train_epoch (np.ndarray): Target data slice for the training set in the current epoch.
    - X_val_epoch (np.ndarray): Feature data slice for the validation set in the current epoch.
    - y_val_epoch (np.ndarray): Target data slice for the validation set in the current epoch.

    Notes:
    - This function performs data slicing based on the provided window size, step size, and epoch number.
    - It ensures the slicing does not go out of bounds by checking the indices before accessing the data.
    - The sliced data for both training and validation sets is used for one epoch of training.
    - If the slicing would exceed the bounds of the input data, `None` is returned for all outputs, signaling the end of the data.

    Example:
    ```python
    X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch = data_slicer(X, y, epoch=0, train_size=3000, val_size=500, step_size=200)
    ```
    """
    # Calculate start and end indices for the data slicing
    train_start = epoch * step_size
    train_end = train_start + train_size
    
    val_start = train_end
    val_end = val_start + val_size
    
    # Ensure we do not go out of bounds
    if val_end > len(X):
        return None, None, None, None
    
    X_train_epoch = X[train_start:val_end]
    y_train_epoch = y[train_start:val_end]

    X_val_epoch = X[val_start:val_end]
    y_val_epoch = y[val_start:val_end]

    return X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch