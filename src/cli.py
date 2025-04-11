import warnings
import argparse
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from data_processor import DataProcessor
from visualizer import Visualizer
from algorithms import CustomTemperaturePredictor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress warnings for clearer output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def main():
    """
    Main function for the Climate Change Analyzer Command Line Interface.
    
    This function parses command line arguments, processes climate data, trains models,
    visualizes data, and analyzes anomalies.
    """
    parser = argparse.ArgumentParser(description="Climate Change Analyzer CLI")
    
    # Define command line arguments
    parser.add_argument('file_path', type=str, help="Path to the CSV file with climate data")
    parser.add_argument('--visualize', action='store_true', help="Visualize the data and generate plots")
    parser.add_argument('--train', action='store_true', help="Train the model and make predictions")
    parser.add_argument('--analyze', action='store_true', help="Analyze anomalies and trends")
    
    args = parser.parse_args()

    # Load and process data
    data_processor = DataProcessor(args.file_path)
    data = data_processor.load_data()
    data_processor.clean_data()
    X, y = data_processor.get_features_and_targets()

    
    # Initialize the linear model
    linear_model = CustomTemperaturePredictor(learning_rate=0.001, n_iterations=2000)
    
    # Create date range for SARIMA model training
    start_date = data_processor.data['DATE'].min()
    end_date = data_processor.data['DATE'].max() - relativedelta(months=1)
    date_series = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Ensure date and target lengths match
    if len(y) != len(date_series):
        raise ValueError(f"Length mismatch: {len(y)} targets vs {len(date_series)} dates")
    
    # Retrieve min and max temperature for normalization
    avg_temp_min = data_processor.data['AVG_TEMP'].min()
    avg_temp_max = data_processor.data['AVG_TEMP'].max()

    # Train the linear model
    linear_model.fit(X, y)
    
    # Make predictions using the trained linear model
    y_pred_linear = linear_model.predict(X)
    y_pred_linear_denorm = y_pred_linear * (avg_temp_max - avg_temp_min) + avg_temp_min
    y_denorm = y * (avg_temp_max - avg_temp_min) + avg_temp_min
    
    # Train the SARIMA model
    linear_model.fit_sarima(date_series, y)
    steps = 32  
    y_pred_sarima = linear_model.predict_sarima(steps=steps)

    raw_anomalies = []
    sarima_anomalies = []

    if args.train:
        # Evaluate model performance
        mae_linear = mean_absolute_error(y_denorm, y_pred_linear_denorm)
        rmse_linear = mean_squared_error(y_denorm, y_pred_linear_denorm)

        mae_sarima = mean_absolute_error(y[-steps:], y_pred_sarima[:steps])
        rmse_sarima = mean_squared_error(y[-steps:], y_pred_sarima[:steps])

        # Print model evaluation metrics
        print("\n--- Model Comparison Results ---")
        print("Linear Model:")
        print(f" - Mean Absolute Error (MAE): {mae_linear:,.2f}")
        print(f" - Mean Squared Error (MSE): {rmse_linear:,.2f}")

        print("\nSARIMA Model:")
        print(f" - Mean Absolute Error (MAE): {mae_sarima:,.2f}")
        print(f" - Mean Squared Error (MSE): {rmse_sarima:,.2f}")

    if args.analyze:
        # Anomaly detection
        print("\n--- Anomaly Detection Analysis ---")
        raw_anomalies = linear_model.detect_anomalies(y, window_size=20, threshold=2.0)
        sarima_anomalies = linear_model.detect_anomalies(y_pred_sarima, window_size=20, threshold=2.0)
        
        print(f"\nDetected {len(raw_anomalies)} anomalies in past data.")
        if raw_anomalies.size:
            print(f"First 5 anomaly indices: {raw_anomalies[:5]}")
        
        print(f"\nDetected {len(sarima_anomalies)} anomalies in future predictions.")
        if sarima_anomalies.size:
            print(f"First 5 predicted anomaly indices: {sarima_anomalies[:5]}")

    if args.visualize:
        # Initialize the Visualizer
        visualizer = Visualizer()

        # Plot Linear vs Actual Data
        visualizer.plot_linear_vs_actual(list(range(len(y))), y, y_pred_linear_denorm)

        # Plot SARIMA vs Actual Data
        visualizer.plot_sarima_vs_actual(list(range(len(y_pred_sarima))), y[-steps:], y_pred_sarima)

        # Plot Future SARIMA Predictions
        visualizer.plot_future_predictions(list(range(len(y), len(y) + steps)), y_pred_sarima, 'Future SARIMA Predictions')


        # Anomaly visualization
        visualizer.plot_anomalies(y, raw_anomalies, 'Anomalies in Linear Model Predictions')
        visualizer.plot_anomalies(y_pred_sarima, sarima_anomalies, 'Anomalies in SARIMA Model Predictions')


        # Clustering
        n_clusters = max(3, int(np.sqrt(len(y) / 2)))
        raw_labels = linear_model.custom_clustering(X, n_clusters)
        sarima_features = np.column_stack((np.arange(len(y_pred_sarima)), y_pred_sarima))
        sarima_labels = linear_model.custom_clustering(sarima_features, n_clusters)

        # Plot clustering results
        visualizer.plot_clustered_data(list(zip(X[:, 0], X[:, 1])), raw_labels, 'Clustering Results (Linear Model)')
        visualizer.plot_clustered_data(list(zip(sarima_features[:, 0], sarima_features[:, 1])), sarima_labels, 'Clustering Results (SARIMA Model)')


    

if __name__ == '__main__':
    main()