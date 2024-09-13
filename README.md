# LSTM-stock-Price-prediction

## Overview
This repository contains a Stock Price Prediction model that forecasts future stock prices based on historical price data. The model leverages time-series analysis techniques using Long Short-Term Memory (LSTM), a type of recurrent neural network (RNN) well-suited for predicting sequential data. This project aims to provide accurate stock price predictions that can be used for financial analysis and decision-making.

## Files
- **stock price prediction.ipynb**: Jupyter Notebook containing the code for basic stock price prediction using historical stock data.
- **stock price prediction 4 parameter.ipynb**: Jupyter Notebook that extends the stock price prediction by considering four key stock-related parameters for enhanced forecasting.
- **README.md**: Documentation and instructions for the project.
- **Google_Stock_train.csv**: The dataset containing historical Google stock price data used for training the model.
- **Google_Stock_test.csv**: The test dataset used to evaluate the model’s performance on unseen data.

## Objective
The goal of this project is to create a predictive model that can forecast future stock prices based on a sequence of past stock prices. This type of model is particularly useful for investors, traders, and financial analysts seeking to gain insights into future market trends and make informed decisions.

## Instructions
### A. Dataset
- The google_stock_train.csv file contains historical Google stock price data, including key columns such as Open, High, Low, Close, Volume, and Adjusted Close.
- The google_stock_test.csv file is used to evaluate the models’ performance on new, unseen data.

### B. Notebook Workflows

**1. stock price prediction.ipynb**

- **Data Preprocessing**: Load and clean the dataset, normalize stock prices, and create time-series sequences for model input.
- **Model Building**: Builds an LSTM model for predicting the next day’s stock prices based on historical data.
- **Training and Evaluation**: Trains the LSTM model and evaluates its accuracy by testing on unseen data.
- **Real-time Prediction**: Once trained, this notebook allows for real-time prediction of future stock prices.

**2. stock price prediction 4 parameter.ipynb**

- **Data Preprocessing**: Similar preprocessing steps but with the incorporation of four important parameters (Open, Close, High, Low) to create more comprehensive input sequences.
- **Model Building**: Builds a more complex LSTM model that considers multiple stock price features to predict the next stock price more accurately.
- **Training and Evaluation**: The model is trained with these four parameters and is evaluated using the test dataset to compare its performance with the simpler model.
- **Enhanced Prediction**: This notebook provides more robust predictions due to the inclusion of multiple key features, leading to better forecasting capabilities.

## Dependencies
To run the project, ensure that the following Python libraries are installed:

- pandas
- numpy
- keras
- tensorflow
- matplotlib
- scikit-learn


## Software Requirements
- **stock_price_prediction.ipynb**: Requires Jupyter Notebook with Python and the necessary libraries installed.
- **historical stock data**: The dataset containing historical stock price data used for training the model.

## Future Work
- **Model Optimization**: Experiment with different architectures, such as GRU or Transformer models, to improve prediction accuracy.
- **Feature Engineering**: Incorporate additional financial indicators (e.g., moving averages, RSI) to enhance the model's predictive capabilities.
- **Real-time Data Integration**: Extend the model to use real-time stock market data for continuous forecasting.
- **Multi-Stock Support**: Expand the model to support predictions for multiple stocks beyond just Google.

## Collaboration Expectations
- Contributions and feedback are welcome through issues and pull requests. 
- Please follow the repository’s contribution guidelines for adding improvements or new features. 

Feel free to contact me for any questions or suggestions!
