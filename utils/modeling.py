"""
This module contains functions for building, training, and tuning a machine learning model using LSTM layers.
The model predicts minimum and maximum percentage changes based on historical data. The module integrates with
Optuna for hyperparameter tuning and includes early stopping, data slicing, and custom callbacks for model training.

Functions:
create_model(input_shape, params_grid) -> keras.models.Sequential
    - Builds and compiles a Sequential LSTM model based on the provided input shape and hyperparameters.
    
data_slicer(X, y, epoch, train_size, val_size, step_size) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    - Slices the data using a rolling window technique, creating training and validation sets for each epoch.
    
optuna_pruning_and_callbacks(trial, model, X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch, params_grid, epoch_counter, early_stopping) -> History
    - Handles Optuna pruning and callbacks during model training, including checks for NaNs and validation loss monitoring.
    
train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=False, trial=None) -> keras.models.Sequential
    - Trains the LSTM model using dynamic data slicing for each epoch and optional Optuna hyperparameter tuning. 
    - Incorporates early stopping to prevent overfitting.
    
Note:
- The module supports LSTM-based time series forecasting with dynamic data slicing for each epoch.
- The `train` function offers flexibility to either perform hyperparameter tuning using Optuna or train the model without it.
- The `optuna_pruning_and_callbacks` function integrates Optuna's pruning mechanism to stop unpromising trials early.
- Early stopping is used to halt training when the validation loss does not improve for a defined number of epochs.

Example usage:

# Train a model without Optuna tuning
model = train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200)

# Train a model with Optuna tuning
model = train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=True, trial=trial)
"""

# Import necessary libraries
import numpy as np
import optuna
import joblib
import tensorflow as tf
from optuna.integration import KerasPruningCallback  
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping
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
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # First LSTM layer with dropout
    model.add(LSTM(params_grid["lstm_units_1"], activation=params_grid["activation"], kernel_initializer=HeNormal(), return_sequences=True))
    model.add(Dropout(params_grid["dropout"]))

    # Second LSTM layer with dropout
    model.add(LSTM(params_grid["lstm_units_2"], activation=params_grid["activation"], kernel_initializer=HeNormal(), return_sequences=False))
    model.add(Dropout(params_grid["dropout"]))

    # Dense layer
    model.add(Dense(params_grid["dense_units"], activation=params_grid["activation"]))

    # Output layer with 2 units for predicted min and max percentage change
    model.add(Dense(2))

    # Select optimizer and compile the model
    if params_grid["optimizer"] == 'adam':
        optimizer = Adam(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])
    else:
        optimizer = RMSprop(learning_rate=params_grid["learning_rate"], clipvalue=params_grid["gradient_clipping"])

    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

def data_slicer(X, y, epoch, train_size, val_size, step_size):
    """
    Slice the data for the current epoch using a rolling window, create targets, and normalize it.

    Parameters:
    - X (np.ndarray): Input feature data.
    - y (np.ndarray): Input target data.
    - window_size (int): Size of the rolling window for feature generation.
    - look_ahead (int): Number of steps to look ahead for the target variable.
    - epoch (int): Current epoch number, used to calculate the starting index for slicing.
    - train_size (int): Number of samples to include in the training set.
    - val_size (int): Number of samples to include in the validation set.
    - step_size (int): The size of the shift for each rolling window (how many steps to move each time).
    
    Returns:
    - X_train_epoch (np.ndarray): Feature data slice for the training set in the current epoch.
    - y_train_epoch (np.ndarray): Target data slice for the training set in the current epoch.
    - X_val_epoch (np.ndarray): Feature data slice for the validation set in the current epoch.
    - y_val_epoch (np.ndarray): Target data slice for the validation set in the current epoch.

    Notes:
    - The function ensures that the slicing does not go out of bounds of the input dataframe.
    - The `create_targets` function is expected to generate feature-target pairs based on the rolling window.
    - The `normalize_y` function is expected to normalize the target values (y) for both training and validation.
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

def optuna_pruning_and_callbacks(trial, model, X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch, params_grid, epoch_counter, early_stopping):
    """
    Handle Optuna pruning and callbacks for model training.
    
    Parameters:
    - trial (optuna.trial.Trial): The current Optuna trial object.
    - model: The model to be trained.
    - X_train_epoch (np.ndarray): Training data for the current epoch.
    - y_train_epoch (np.ndarray): Target data for the current epoch.
    - X_val_epoch (np.ndarray): Validation data for the current epoch.
    - y_val_epoch (np.ndarray): Target data for validation.
    - params_grid (dict): Dictionary of hyperparameters for model.
    - epoch_counter (int): The current epoch counter.
    - early_stopping (EarlyStopping): Early stopping callback.
    
    Returns:
    - history: Training history of the current epoch.
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

    # Use Optuna's KerasPruningCallback to prune the trial if necessary
    #pruning_callback = KerasPruningCallback(trial, 'val_loss')  # Monitor 'val_loss' for pruning

    # Now train the same model on the new data for this epoch (do not reinitialize model)
    history = model.fit(
        X_train_epoch,
        y_train_epoch,
        epochs=1,  # Train for 1 epoch at a time in this loop
        batch_size=params_grid["batch_size"],
        validation_data=(X_val_epoch, y_val_epoch),  # Provide the new validation data
        verbose=1,
        callbacks=[early_stopping, nan_checker] #, pruning_callback]
    )

    # Check if Optuna should prune the trial
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return history


def train(X, y, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=False, trial=None):
    """
    Train the model with early stopping, data slicing, and optional Optuna hyperparameter tuning.
    Parameters:
    - model_df (pd.DataFrame): Input dataframe containing the training and validation data.
    - params (dict): Dictionary containing the model parameters, such as 'window_size', 'num_indicators', etc.
    - epochs (int): Total number of epochs to train. Default is 50.
    - train_size (int): Number of training samples per epoch. Default is 3000.
    - val_size (int): Number of validation samples per epoch. Default is 500.
    - step_size (int): The size of the shift for each rolling window in the data. Default is 200.
    - tuning (bool): Flag to indicate if Optuna hyperparameter tuning should be applied. Default is False.
    - trial (optuna.trial.Trial, optional): The Optuna trial object to track the tuning process if `tuning=True`.

    Returns:
    - model: The trained model after completing the specified number of epochs or early stopping.
    
    Notes:
    - The function supports both training with and without Optuna-based hyperparameter tuning.
    - Uses early stopping to prevent overfitting by monitoring validation loss and restoring the best model weights.
    - The `data_slicer` function slices the dataset into training and validation sets for each epoch using a rolling window.
    - The model is trained for one epoch at a time to update the training and validation data dynamically.
    - If early stopping is triggered, training is stopped prematurely.

    Example:
    ```python
    model = train(model_df, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=True, trial=trial)
    ```
    """
    
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
    for epoch in range(epochs):
        # Get the new training and validation data slices for the current epoch
        X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch = data_slicer(X, y, epoch, train_size, val_size, step_size)
        
        if X_train_epoch is None:
            print("Reached end of data, stopping training.")
            break
        
        if tuning:
            # Call Optuna-specific logic if tuning is enabled
            history = optuna_pruning_and_callbacks(trial, model, X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch, params_grid, epoch_counter, early_stopping)
        else:
            # Train the model without Optuna
            history = model.fit(
                X_train_epoch,
                y_train_epoch,
                epochs=1,  # Train for 1 epoch at a time in this loop
                batch_size=params_grid["batch_size"],
                validation_data=(X_val_epoch, y_val_epoch),  # Provide the new validation data
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
    norm_X_test, y_test = create_targets(test_df, params['window_size'], params['look_ahead_size'], params['look_ahead_size'])
    norm_y_test = normalize_y(y_test)
    
    if model_name is not None:
        joblib.dump(norm_X_test, f'../models/{model_name}/X_normTest.pkl')
        joblib.dump(norm_y_test, f'../models/{model_name}/y_normTest.pkl')
        joblib.dump(y_test, f'../models/{model_name}/y_realTest.pkl')
        print('Test data dumped')
    
    return norm_X_test, norm_y_test, y_test

def custom_model(df, params, model_name=None, tuning=False, trial=None):
    """
    Trains a custom model using the provided dataframe and parameters.
    
    Parameters:
        df (pd.DataFrame): The input dataframe containing the data.
        params (dict): A dictionary of parameters for the model, including:
            - 'scaler_type' (str): The type of scaler to use for normalization.
            - 'window_size' (int): The size of the window for creating features.
            - 'look_ahead_size' (int): The size of the look-ahead window for creating targets.
        model_name (str, optional): The name of the model for saving the scaler. Defaults to None.
        tuning (bool, optional): Whether the model is being tuned. Defaults to False.
        trial (optuna.trial.Trial, optional): The trial object for hyperparameter tuning. Defaults to None.
    Returns:
        - model: The trained model.
        - scaler_y: The scaler used to normalize the target
    """

    # Load data and create indicators
    model_df = normalize_X(df, scaler_name=params['scaler_type'])
    X_train, y_train = create_targets(model_df, params['window_size'], params['look_ahead_size'], params['look_ahead_size'])
    
    y_train, scaler_y = normalize_y(y_train, return_scaler=True)
    
    if model_name is not None:
        joblib.dump(scaler_y, f'../models/{model_name}/scaler_y.pkl')
        print('Scaler dumped')
    
    # Train
    model = train(X_train, y_train, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=tuning, trial=trial) 
    
    return model, scaler_y