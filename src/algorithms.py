# Innovative Algorithms

# IMPORTS
import numpy as np
import pandas as pd
import data_processor
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Tuple


# Class used for machine learning algorithms and functions. 
class CustomTemperaturePredictor(BaseEstimator, RegressorMixin):

    # Constructor for the class, uses a default float of 0.001 for learning rate & 2000 iterations. 
    # This is to ensure the models are properly trained
    def __init__(self, learning_rate: float = 0.01, n_iterations: int= 2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.slope = 0.0
        self.intercept = 0.0
        self.bias = 0.0

        # Written to initialize the SARIMA Model
        # For the SARIMA Model, uses months as its seasonal order
        self.sarima_model = None
        self.sarima_order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 12)


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CustomTemperaturePredictor':
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X should be 2D and y should be 1D")
        # Extracts the samples and features of the dataset using shape
        m, n = X.shape
        
        # Random weights given for the training step, bias set to 0
        self.weights = np.zeros(n)
        self.bias = 0

        # Uses L2 regularization to ensure more accuracy for our linear model
        lambda_reg = 0.1

        # Code for training the model, uses a gradient to do so
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            # Formula for the gradients, with L2 regularization included
            gradient_weights = (-2/m) * np.dot(X.T, error) + lambda_reg * self.weights
            gradient_bias = (-2/m) * np.sum(error)
            
            # Ensure the data set does not contain NaN values
            if np.isnan(gradient_weights).any() or np.isnan(gradient_bias):
                print("NaN detected. Training halted")
                break
           
            # Clips the gradient weights and bias to help with normalization of the data
            gradient_weights = np.clip(gradient_weights, -1.0, 1.0)
            gradient_bias = np.clip(gradient_bias, -1.0, 1.0)

            # Sets the weights and biases for the linear model
            self.weights -= self.learning_rate * gradient_weights
            self.bias -= self.learning_rate * gradient_bias

        return self
    
    # Function used to fit the SARIMA model for our weather data
    def fit_sarima(self, date_series: pd.Series, y: np.ndarray) -> None:

        # Use the date_series variable as the predictor for the SARIMA model
        df = pd.DataFrame({'Date': date_series, 'Temp': y})
        df.set_index('Date', inplace=True)
        
        # Use the SARIMAX function to fit the SARIMA model with seasonal order
        self.sarima_model = SARIMAX(
                df['Temp'], order=self.sarima_order, seasonal_order=self.seasonal_order, 
                enforce_stationary=False, enforce_invertibility=False).fit(disp=False)


    # Predictor for the linear model, which returns a dot product of the trained data
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias

    # Predictor for the SARIMA model
    # Checks that the model has been trained, and creates a forecast with the given # of steps
    def predict_sarima(self, steps: int) -> np.ndarray:
        if self.sarima_model is None:
            raise ValueError("Sarima model is not trained yet.")

        forecast = self.sarima_model.forecast(steps=steps)
        return forecast.values

    # Clustering algorithm for both the Linear and SARIMA models
    def custom_clustering(self, data: np.ndarray, n_clusters: int) ->np.ndarray:
        
        # We set the number of clusters based on the size of our dataset (n = 277)
        # The number of clusters is equal to the square root of n/2 (about 11-12)
        if n_clusters is None:
            n_clusters = max(3, int(np.sqrt(len(data) / 2)))
        

        # uses a random seed of 42, and then finds the centroids within the clustered data
        np.random.seed(42)
        centroids = data[np.random.choice(len(data), n_clusters, replace=False)]

        # Uses 15 iterations to ensure stability of the data
        # Finds the centroids of each cluster and the distances between them
        # And returns the labels of the clustering algorithm
        for _ in range(15):
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])

        return labels

    # Detects anomalies in the data, uses a window size of 20, and z-score threshold of 2.0
    def detect_anomalies(self, time_series: np.ndarray, window_size: int = 20, threshold: float = 2.0) -> np.ndarray:

        # Initialize an empty list of anomalies
        anomalies = []

        # Find the mean and standard deviation of our time series data
        mean = np.convolve(time_series, np.ones(window_size)/window_size, mode='valid')
        std = np.std(time_series)

        # If the z-score of a value is above the threshold, it's added to the list of anomalies.
        for i, value in enumerate(time_series[window_size-1:]):
            z_score = (value - mean[i]) / std
            if np.abs(z_score) > threshold:
                anomalies.append(i + window_size - 1)
        
        return np.array(anomalies)




