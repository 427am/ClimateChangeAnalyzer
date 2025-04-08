# Main

# IMPORTS
from data_processor import DataProcessor
from algorithms import CustomTemperaturePredictor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import os
from dateutil.relativedelta import relativedelta


def main():
    # Load and preprocess data
    # Uses os to locate our csv for WeatherData
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'data', 'WeatherData.csv')

    # Checks to ensure the file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Uses the DataProcessor class to load and clean our dataset, as well as get features & targets
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    X, y = processor.get_features_and_targets()

    # Find the start and end dates for our dataset
    # relativedelta used to ensure the # of targets is equal to the dataset size
    start_date = processor.data['DATE'].min()
    end_date = processor.data['DATE'].max() - relativedelta(months=1)

    # date_series is a variable used for the SARIMA model, uses months
    date_series = pd.date_range(start=start_date, end=end_date, freq='M')

    # Checks that the number of targets is equal to the dataset size
    if len(y) != len(date_series):
        raise ValueError(f"Length mismatch: {len(y)} targets vs {len(date_series)} dates")
    
    # find the min and max for average temp to help train the linear model
    avg_temp_min = processor.data['AVG_TEMP'].min()
    avg_temp_max = processor.data['AVG_TEMP'].max()
    
    # Temperature prediction and fitting for the linear model
    linear_model = CustomTemperaturePredictor(learning_rate=0.001, n_iterations=2000)
    linear_model.fit(X, y)

    # Data is slightly denormalized to ensure the linear model is better comparable to SARIMA
    y_pred_linear = linear_model.predict(X)
    y_pred_linear_denorm = y_pred_linear * (avg_temp_max - avg_temp_min) + avg_temp_min
    y_denorm = y * (avg_temp_max - avg_temp_min) + avg_temp_min
    
    # Temperature prediction with SARIMA model
    linear_model.fit_sarima(date_series, y)
    steps = 32

    # SARIMA is trained for future data, which the following code reflects
    future_data = pd.date_range(start=date_series[-1], periods=steps + 1, freq='M')[1:]
    y_pred_sarima = linear_model.predict_sarima(steps=steps)

    # Clustering
    # uses n_clusters value equal to the square root of n/2
    n_clusters = max(3, int(np.sqrt(len(y) / 2)))
    raw_labels = linear_model.custom_clustering(X, n_clusters)

    # Features and labels for clustering of the SARIMA model
    sarima_features = np.column_stack((np.arange(len(y_pred_sarima)), y_pred_sarima))
    sarima_labels  = linear_model.custom_clustering(sarima_features, n_clusters)

    # Anomaly Detection
    # Uses the detect_anomalies algorithm for both the Linear and SARIMA models, z-score of 2.0
    raw_anomalies = linear_model.detect_anomalies(y, window_size=20, threshold=2.0)
    sarima_anomalies = linear_model.detect_anomalies(y_pred_sarima, window_size=20, threshold=2.0)

    # Evaluate accuracy of each model
    # Find the Mean Absolute and Mean Squared Errors for our linear model
    mae_linear = mean_absolute_error(y_denorm, y_pred_linear_denorm)
    rmse_linear = mean_squared_error(y_denorm, y_pred_linear_denorm)

    # Find the Mean Absolute and Mean Squared Errors for our SARIMA model
    y_sarima_sample = linear_model.sarima_model.fittedvalues
    mae_sarima = mean_absolute_error(y[-steps:], y_pred_sarima[:steps])
    rmse_sarima = mean_squared_error(y[-steps:], y_pred_sarima[:steps])

    # Print the results
    print("\n Comparing Accuracy of Linear and SARIMA Models:")
    print(f"Linear Model - MAE: {mae_linear}, MSE: {rmse_linear}")
    print(f"SARIMA Model - MAE: {mae_sarima}, MSE: {rmse_sarima}")
    
    pass


if __name__ == "__main__":
    main()
