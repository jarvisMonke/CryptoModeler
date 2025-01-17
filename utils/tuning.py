import os
import json
import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from utils.modeling import load_model_test_split, test_data, custom_model

# Objective function for optimization
def objective(trial, config):
    """
Objective function for hyperparameter tuning using Optuna.

This function defines the hyperparameters to be tuned and the model training 
process. It returns the combined root mean squared error (RMSE) of the model 
predictions on the test set. The goal is to optimize the hyperparameters 
to improve the model's performance on financial time series prediction tasks.

Parameters:
    trial (optuna.trial.Trial): A trial object that suggests hyperparameters during optimization.
    config (dict): A dictionary containing configuration values for the model, including:
        - window_size (int): Size of the input sequence window.
        - look_ahead_size (int): Number of time steps to look ahead for predictions.
        - crypto (str): The cryptocurrency symbol (e.g., "BTC").
        - timeframe (str): Timeframe for the data (e.g., '1m', '3m', '5m', etc.).
        - timeframe_filenames (str or list): Filenames or paths for the time series data.

Returns:
    float: The combined RMSE (Root Mean Squared Error) of the model's predictions on the test set.

Hyperparameters Tuned:
    - lstm_units_1 (int): Number of units in the first LSTM layer.
    - lstm_units_2 (int): Number of units in the second LSTM layer.
    - dropout (float): Dropout rate for regularization.
    - learning_rate (float): Learning rate for the optimizer.
    - batch_size (int): Batch size for training.
    - dense_units (int): Number of units in the dense layer after the LSTM layers.
    - sequence_length (int): Length of the input sequences.
    - gradient_clipping (float): Value for gradient clipping to prevent exploding gradients.
    - optimizer (str): Optimizer type ('adam' or 'rmsprop').
    - activation (str): Activation function for layers ('relu' or 'tanh').
    - num_indicators (int): Number of technical indicators used in the model.
    - scaler_type (str): Type of scaler used for data normalization ('MinMaxScaler', 'StandardScaler', or 'RobustScaler').
    - timeframe (str): Timeframe for the data (e.g., '1m', '3m', etc.).
    - window_size (int): Size of the input sequence window.
    - look_ahead_size (int): Number of time steps to look ahead for predictions.

Raises:
    optuna.exceptions.TrialPruned: If the trial is pruned due to early stopping or poor performance during training.

Usage:
    This function is used in the Optuna optimization loop to find the best hyperparameters 
    for a machine learning model that predicts financial market data. The objective is 
    to minimize the RMSE of the model's predictions, thereby improving its ability to 
    forecast future price movements.

Example:
    # Example usage in Optuna optimization loop
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, config), n_trials=100)
    
Note:
    - Ensure that the required data files (e.g., historical price data) are available and properly 
      referenced in `timeframe_filenames`.
    - The model's hyperparameters are tuned dynamically using Optuna's `trial.suggest_*` methods.
    - The function returns the combined RMSE as a metric for optimization, where a lower RMSE indicates better performance.
"""

    # MODEL CREATION
    params = {
        # LSTM Units
        "lstm_units_1": trial.suggest_categorical("lstm_units_1", [64, 128, 256]),
        "lstm_units_2": trial.suggest_categorical("lstm_units_2", [32, 64, 128]),

        # Dropout rate for regularization
        "dropout": trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.5]),

        # Learning rate for the optimizer (log scale for better range)
        "learning_rate": trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),

        # Batch size (common choices)
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),

        # Dense layer units (for fully connected layers after LSTM)
        "dense_units": trial.suggest_categorical("dense_units", [16, 32, 64, 128]),

        # Sequence length (historical data considered for prediction)
        "sequence_length": trial.suggest_categorical("sequence_length", [10, 20, 50, 100]),

        # Gradient clipping for preventing exploding gradients
        "gradient_clipping": trial.suggest_categorical("gradient_clipping", [0.5, 1.0, 2.0]),

        # Optimizer choice (Adam or RMSprop)
        "optimizer": trial.suggest_categorical("optimizer", ['adam', 'rmsprop']),

        # Activation function for layers
        "activation": trial.suggest_categorical("activation", ['relu', 'tanh']),

        "num_indicators" : trial.suggest_int('num_indicators', 5, 46),

        "scaler_type" : trial.suggest_categorical('scaler_type', ['MinMaxScaler', 'StandardScaler', 'RobustScaler']),
 
        'window_size' : config.get("window_size", trial.suggest_categorical('window_size', [20, 40, 60, 80, 100])),
        'look_ahead_size' : config.get("look_ahead_size", 20),
        'window_shift': config.get("window_shift", trial.suggest_categorical('window_shift', [1, 5, 10, 15, 20])),

        'crypto' : config.get("crypto", "ETH"),
        'timeframe' : config.get("timeframe", "1m"),
        'timeframe_filenames' : config.get("timeframe_filenames"),
        'epochs' : config.get("epochs", 50),

        'train_size' : config.get("train_size", trial.suggest_categorical('train_size', [1000, 2000, 3000, 5000, 10000, 20000])),
        'val_size' : config.get("val_size", 500),
        'step_size' : config.get("step_size", trial.suggest_categorical('step_size',[100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000])),
        
        'shuffle' : config.get("shuffle", True)
    }

    model_df, test_df = load_model_test_split(params['timeframe_filenames'], params)

    print(f'Trial Hyperparams: {params}')
    model, scaler_y = custom_model(model_df, params, tuning=True, trial=trial)

    # Check if the trial is pruned after training
    if trial.should_prune():
        print("Trial pruned after training.")
        raise optuna.exceptions.TrialPruned()
    
    X_test, Y_test, y_real = test_data(test_df, params)
    
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)

    rmse_max = np.sqrt(mean_squared_error(y_real[:,0], y_pred[:,0]))
    rmse_min = np.sqrt(mean_squared_error(y_real[:,1], y_pred[:,1]))
    rmse_combined = (rmse_min + rmse_max) / 2
    
    return rmse_combined 


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

def manage_optuna_study(study_name, config, delete_existing=False):
    """
Manages the creation, deletion, and loading of an Optuna study.

This function handles the setup of an Optuna study by creating a new study if it doesn't exist, 
loading an existing study if available, and optionally deleting any previous study data and configuration.

Parameters:
    study_name (str): The name of the study, which will be used to create/load the study's database and configuration.
    config (dict): The configuration dictionary for the study. It will be saved if the study is created.
    delete_existing (bool): A flag indicating whether to delete existing study data and configuration.
                             If set to `True`, any existing study database and configuration will be removed.

Returns:
    tuple: A tuple containing two elements:
        - study (optuna.study.Study): The Optuna study object.
        - config (dict): The configuration dictionary for the study.

Raises:
    FileNotFoundError: If the configuration file exists but cannot be opened.
    OSError: If there is an error creating directories or deleting files.

Usage:
    This function is useful for managing the lifecycle of an Optuna study. It can be used to create a new study,
    load an existing study, and handle the deletion of existing data. The study name and configuration are
    stored on the filesystem, and the study object can be used to perform optimization tasks.

Example:
    # Create a new study or load an existing one
    study_name = "hyperparameter_optimization"
    config = {"learning_rate": 0.01, "batch_size": 32}
    study, config = manage_optuna_study(study_name, config, delete_existing=False)

    # Delete an existing study and its configuration
    study, config = manage_optuna_study(study_name, config, delete_existing=True)
"""

    study_path = f'../optuna/{study_name}/study.db'
    storage_url = f'sqlite:///{study_path}'
    config_path = f'../optuna/{study_name}/config.json'
    
    os.makedirs(os.path.dirname(f'../optuna/{study_name}/'), exist_ok=True)

    # Handle study deletion if required
    if delete_existing:
        if os.path.exists(study_path):
            os.remove(study_path)
            print(f"Deleted existing study '{study_name}' at {study_path}")
        if os.path.exists(config_path):
            os.remove(config_path)
            print(f"Deleted existing configuration for '{study_name}' at {config_path}")

    # Load or create the study
    if os.path.exists(study_path) and not delete_existing:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        print(f"Loaded existing study '{study_name}' from {study_path}")
        
        # Load the existing configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded existing configuration for study '{study_name}':")
        print(json.dumps(config, indent=4))
    else:
        # Create a new study
        study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize")
        print(f"Created new study '{study_name}' at {study_path}")
        
        # Save the new configuration to a JSON file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved new configuration for study '{study_name}':")
        print(json.dumps(config, indent=4))
    
    return study, config


