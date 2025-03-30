import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from algorithms import CustomTemperaturePredictor
from statsmodels.tsa.statespace.sarimax import SARIMAX

def test_linear_model():
    # Create mock training and testing sets for weather data
    X_train = np.array([[2010, 1], [2011, 2], [2012, 3], [2013, 4], [2014, 5], 
                        [2015, 6], [2016, 7], [2017, 8], [2018, 9], [2019, 10]])

    y_train = np.array([65, 72, 75, 77, 73, 81, 79, 75, 77, 72,])
    X_test = np.array([[2020, 5], [2021, 6], [2022, 7]])
    y_test = np.array([76, 74, 83])

    # Create and fit the model
    model = CustomTemperaturePredictor()

    model.fit(X_train, y_train)
    
    # Ensure each model has a slope and intercept
    assert model.slope is not None, "Slope should not be 0."
    assert model.intercept is not None, "Model should contain an intercept."

    # Create the predictions array
    y_pred = model.predict(X_test)

    # Assert the number of predictions is equal to the testing set
    assert len(y_pred) == len(X_test), "# of predictions must be equal to # of inputs."

def test_sarima_model():
    # Create a mock dataset for the SARIMA model
    date_series = pd.date_range(start='2000-01-01', periods=5, freq='A')
    y = np.array([30, 32, 33, 35, 36])

    # Create and fit the SARIMA Model
    model = CustomTemperaturePredictor(learning_rate=0.01, n_iterations=2000)

    model.fit_sarima(date_series, y)

    # Use the SARIMA model to create future predictions
    steps = 32
    future_predictions = model.predict_sarima(steps=steps)

    # Test that the size of the future predictions and the data types are correct
    assert len(future_predictions) == steps, "Mismatch in count of predictions."
    assert isinstance(future_predictions, np.ndarray), "Predictions should be a numpy array."

def test_custom_clustering():
    # Mock dataset used to test clustering
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4]])

    # Create a model and two clusters to support it
    model = CustomTemperaturePredictor()

    labels = model.custom_clustering(data, n_clusters=2)
    
    # Assert the correct number of clusters, and size of the label set
    assert len(np.unique(labels)) == 2, "Clustering should produce 2 clusters."
    assert len(labels) == len(data), "Labels should match the data size."

def test_anomaly_detection():
    # Create a mock dataset to detect anomalies
    anomaly_data = np.array([10, 12, 13, 15, 100, 18, 20, 22, 25, 30])

    # Create the model
    model = CustomTemperaturePredictor()

    # Find the anomalies (above a z-score threshold of 2)
    anomalies = model.detect_anomalies(anomaly_data, window_size=3, threshold=2.0)

    # Assert at least one anomaly was detected, and that it is of the correct index
    assert len(anomalies) > 0, "No anomalies detected."
    assert anomalies[0] == 4, "Anomaly detection did not detect the correct outlier."

if __name__ == '__main__':
    pytest.main()
