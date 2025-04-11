# VISUALIZER
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple
import pandas as pd
import matplotlib.dates as mdates

class Visualizer:
    """
    A class for visualizing temperature trends, model predictions, clusters, and anomalies.
    """

    @staticmethod
    def plot_temperature_trend(years: List[int], temperatures: List[float], predictions: List[float], title: str) -> None:
        """
        Plot the actual vs predicted temperature trends over time.
        :param years: List of years (feature).
        :param temperatures: List of actual temperature values (target).
        :param predictions: List of predicted temperature values (from the model).
        :param title: Title for the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(years, temperatures, label='Actual', marker='o', color='blue')
        plt.plot(years, predictions, label='Predicted', linestyle='--', marker='x', color='red')
        plt.xlabel('Year')
        plt.ylabel('Temperature (normalized)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_future_predictions(future_years: List[int], predictions: List[float], title: str) -> None:
        """
        Plot future predictions (SARIMA model).
        :param future_years: List of future years.
        :param predictions: List of predicted temperature values for future years.
        :param title: Title for the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(future_years, predictions, label='Future Predictions', marker='o', color='green')
        plt.xlabel('Year')
        plt.ylabel('Temperature (normalized)')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_clustered_data(data: List[Tuple[float, float]], labels: List[int], title: str) -> None:
        """
        Plot clustered data points with different colors for each cluster.
        :param data: List of (x, y) data points.
        :param labels: Cluster labels for each data point.
        :param title: Title for the plot.
        """
        x, y = zip(*data)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, hue=labels, palette='Set2', s=60, edgecolor='k')
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('cluster size')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()
  
    @staticmethod
    def plot_anomalies(dates, values, anomalies, title):
        """
        Plots data with highlighted anomalies.

        Parameters:
        - dates: list or pandas Series of datetime objects (for x-axis)
        - values: list or array of temperature values (denormalized if desired)
        - anomalies: list of booleans indicating which points are anomalies
        - title: string for the plot title
        """
        anomaly_indices = [i for i, is_anomaly in enumerate(anomalies) if is_anomaly]
        anomaly_dates = [dates[i] for i in anomaly_indices]
        anomaly_values = [values[i] for i in anomaly_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, label='Temperature')
        plt.scatter(anomaly_dates, anomaly_values, color='red', label='Anomalies', zorder=5)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_sarima_vs_actual(years: List[int], actuals: List[float], sarima_predictions: List[float]) -> None:
        """
        Plot SARIMA model predictions vs actual data.
        :param years: List of years.
        :param actuals: List of actual temperature values.
        :param sarima_predictions: List of SARIMA predictions.
        """
        Visualizer.plot_temperature_trend(years, actuals, sarima_predictions, 'SARIMA vs Actual Temperature')

    @staticmethod
    def plot_linear_vs_actual(years: List[int], actuals: List[float], linear_predictions: List[float]) -> None:
        """
        Plot Linear model predictions vs actual data.
        :param years: List of years.
        :param actuals: List of actual temperature values.
        :param linear_predictions: List of Linear model predictions.
        """
        Visualizer.plot_temperature_trend(years, actuals, linear_predictions, 'Linear Model vs Actual Temperature')

