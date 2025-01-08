"""
This module contains various functions for predicting, analyzing, and visualizing stock market data. 
It provides functionalities to load a trained model, prepare input data, make predictions, visualize distributions, 
simulate profits based on predicted and actual values, and calculate the percentage of values above a threshold.

Functions:
- predict(model_name, ochlv): Loads a model and its parameters, preprocesses the input data, makes a prediction, and returns the descaled result.
- distribution_plot(*data_sets, bins=30, colors=None): Plots the distribution of multiple datasets with Kernel Density Estimate (KDE).
- profit_simulation(y_pred, y_real, threshold=0.008): Simulates the profit based on predicted and actual values using a specified threshold.
- calculate_percentage_above_threshold(y_test, threshold=0.01): Calculates the percentage of values above a given threshold in the test data.

Dependencies:
- numpy: For numerical computations and handling data arrays.
- seaborn: For visualizing the data distributions using histograms and KDE.
- matplotlib: For creating visual plots.
- utils.preprocess: For creating technical indicators from stock data.
- utils.modeling: For normalizing the input data.
- utils.helpers: For loading the trained model and associated parameters.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocess import create_indicators 
from utils.modeling import normalize_X
from utils.helpers import load_model

def predict(model_name, ochlv):
    """
    Predicts the target value using a trained model.

    This function loads a previously saved model along with its associated parameters 
    and scalers. It then prepares the input data by generating indicators, normalizing it,
    and formatting it into a shape suitable for the model. The function makes a prediction 
    and returns the descaled prediction to provide the result in the original scale.

    Parameters:
    model_name (str): The name of the saved model to load.
    ochlv (DataFrame): A pandas DataFrame containing the open, close, high, low, and volume 
                        (OCHLV) data used for generating indicators and making predictions.

    Returns:
    numpy.ndarray: The descaled prediction returned by the model.
    """

    model, params, scaler_X, scaler_y, X_normTest, y_normTest, y_realTest = load_model(model_name)
    # Generate indicators and prepare the data for prediction

    indicated_data = create_indicators(ochlv, params['num_indicators'])[-params['window_size']:]
    X_data = normalize_X(indicated_data, scaler_name=params['scaler_type'], specific_scaler=scaler_X)
    X_data.drop('close', axis=1, inplace=True)
    X_sample = np.expand_dims(X_data, axis=0)

    # Predict using the model
    prediction = model.predict(X_sample)
    
    # Inverse transform the prediction to return the original scale
    descaled_prediction = scaler_y.inverse_transform(prediction)

    return descaled_prediction

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

def profit_simulation(y_pred, y_real, threshold=0.008):
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
    for pred, actual in zip(y_pred, y_real):
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

def calculate_percentage_above_threshold(y_test, threshold=0.01):
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