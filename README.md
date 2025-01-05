# Cryptocurrency Price Prediction Model Generator

This repository contains a machine learning-powered system designed to generate models for predicting cryptocurrency price movements based on historical data. The system is in development and currently focuses on a single coin. It uses machine learning algorithms to create models that aim to predict future price trends and generate insights.

## Project Overview

The system is being developed to generate machine learning models that predict the future price movements of a single cryptocurrency. It uses LSTM (Long Short-Term Memory) networks to analyze historical data and generate predictive models. These models can later be used for making automated trading decisions or analyzing market trends.

### Technologies Used:
- **Python 3.x**
- **TensorFlow / Keras** (for model building and training)
- **Optuna** (for hyperparameter optimization)
- **pandas** (for data handling)
- **NumPy** (for numerical computations)
- **Matplotlib / Plotly** (for visualization)
- **TA-Lib / Custom Indicators** (for technical analysis)
- **API Integration** (for future live trading applications)

## Features

- **Single Coin Focus**: Currently, the system is designed to generate models for a single cryptocurrency coin.
- **LSTM-Based Model**: Uses deep learning models (LSTM) to predict cryptocurrency price movements.
- **Dynamic Hyperparameter Optimization**: Uses **Optuna** for tuning model hyperparameters, such as the number of LSTM units, dropout rates, batch sizes, and learning rates.
- **Profit Maximization**: Aims to create models that are optimized for profit generation when applied to real-world trading.
- **Early Stopping and Pruning**: Implements **early stopping** and **pruning** based on validation loss to save computational resources during training.
- **Custom Indicators**: Supports using a wide range of **technical analysis indicators** (RSI, SMA, EMA, etc.) to enhance the model's prediction capabilities.

## Current Status

This project is in development and focuses on creating models to predict price movements for a single cryptocurrency. The core model-building components are in place, but further work is being done to enhance model accuracy, performance, and optimization. Contributions and feedback are welcome!

## Future Enhancements

- Expand model generation to support multiple cryptocurrencies
- Integrate advanced machine learning techniques for model improvement
- Implement additional technical analysis indicators for enhanced predictions
- Transition to real-time model deployment for automated trading systems