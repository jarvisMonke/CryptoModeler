import numpy as np
import optuna
import tensorflow as tf
from optuna.integration import KerasPruningCallback  
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.callbacks import EarlyStopping
from utils.preprocess import normalize_X, normalize_y, create_targets

# Set the random seed for TensorFlow
tf.random.set_seed(42)

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

def data_slicer(df, window_size, look_ahead, epoch, train_size, val_size, step_size):
    """
    Slice the data for the current epoch using a rolling window and normalize it.
    
    Parameters:
    - X_model (np.ndarray): Input data for training.
    - y_model (np.ndarray): Target data for training.
    - epoch (int): Current epoch number.
    - train_size (int): Number of training samples.
    - val_size (int): Number of validation samples.
    - step_size (int): The size of the shift for each rolling window.
    
    Returns:
    - X_train_epoch (np.ndarray): Training data slice for the epoch.
    - y_train_epoch (np.ndarray): Target data slice for training.
    - X_val_epoch (np.ndarray): Validation data slice for the epoch.
    - y_val_epoch (np.ndarray): Target data slice for validation.
    """
    # Calculate start and end indices for the data slicing
    train_start = epoch * step_size
    train_end = train_start + train_size
    
    val_start = train_end
    val_end = val_start + val_size
    
    # Ensure we do not go out of bounds
    if val_end > len(df):
        return None, None, None, None
    
    train_epoch = df[train_start:val_end]
    val_epoch = df[val_start:val_end]

    X_train_epoch, y_train_epoch = create_targets(train_epoch, window_size, look_ahead, look_ahead)
    X_val_epoch, y_val_epoch = create_targets(val_epoch, window_size, look_ahead, look_ahead)
    
    # Normalize the sliced data
    y_train_epoch, y_val_epoch = normalize_y(y_train_epoch, y_val_epoch)
    
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
    pruning_callback = KerasPruningCallback(trial, 'val_loss')  # Monitor 'val_loss' for pruning

    # Now train the same model on the new data for this epoch (do not reinitialize model)
    history = model.fit(
        X_train_epoch,
        y_train_epoch,
        epochs=1,  # Train for 1 epoch at a time in this loop
        batch_size=params_grid["batch_size"],
        validation_data=(X_val_epoch, y_val_epoch),  # Provide the new validation data
        verbose=1,
        callbacks=[early_stopping, nan_checker, pruning_callback]
    )
    
    # Manually increment the Optuna pruning step
    trial.report(history.history['val_loss'][-1], step=epoch_counter)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return history


def train(model_df, params, epochs=50, train_size=3000, val_size=500, step_size=200, tuning=False, trial=None):
    """
    Train the model using early stopping and NaN checking.
    
    Parameters:
    - X_model (np.ndarray): Input data for training.
    - y_model (np.ndarray): Target data for training.
    - params_grid (dict): Dictionary of hyperparameters for model.
    - epochs (int): Total number of epochs to train.
    - train_size (int): The size of the training data for each epoch.
    - val_size (int): The size of the validation data for each epoch.
    - step_size (int): The rolling window shift size for each epoch.
    - tuning (bool): Flag to indicate if Optuna tuning is used.
    - trial (optuna.trial.Trial): The Optuna trial to tune (optional).
    
    Returns:
    - model: Trained model.
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
        X_train_epoch, y_train_epoch, X_val_epoch, y_val_epoch = data_slicer(model_df, params['window_size'], params['look_ahead_size'], epoch, train_size, val_size, step_size)
        
        if X_train_epoch is None:
            print("Reached end of data, stopping training.")
            break
        
        if tuning and trial is not None:
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

