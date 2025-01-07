import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from utils.modeling import load_model_test_split, test_data, custom_model

# Objective function for optimization
def objective(trial):

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
 
        'timeframe': '1m', 
        'window_size': 80, 
        'look_ahead_size': 20
    }

    # DATA CREATION AND PROCESSING
    timeframe_filenames = {
    "1m": "../data/raw/ETHUSDT_1m_v600k.csv",
    "3m": "../data/raw/DOGEUSDT_3m_v0.csv",
    "5m": "../data/raw/DOGEUSDT_5m_v0.csv",
    "15m": "../data/raw/DOGEUSDT_15m_v0.csv",
    "30m": "../data/raw/DOGEUSDT_30m_v0.csv"
}

    model_df, test_df = load_model_test_split(timeframe_filenames, params)

    print(params)
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