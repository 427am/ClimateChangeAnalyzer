# Data Visualization

# IMPORTS
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_temperature_trend(years: List[int], temperatures: List[float], predictions: List[float]) -> None:
        plt.figure(figsize=(10, 6))  # Adjusted figure size
        plt.plot(years, temperatures, label='Actual', marker='o')
        plt.plot(years, predictions, label='Predicted', linestyle='--', marker='x')
        plt.xlabel('Year')
        plt.ylabel('Temperature (normalized)')
        plt.title('Temperature Trend Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_clustered_data(data: List[Tuple[float, float]], labels: List[int]) -> None:
        x, y = zip(*data)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, hue=labels, palette='Set2', s=60, edgecolor='k')
        plt.title('Clustered Data Visualization')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_anomalies(time_series: List[float], anomalies: List[bool]) -> None:
        plt.figure(figsize=(10, 6))
        time = list(range(len(time_series)))
        plt.plot(time, time_series, label='Time Series')
        
        anomaly_points = [val if is_anomaly else None for val, is_anomaly in zip(time_series, anomalies)]
        plt.scatter(time, anomaly_points, color='red', label='Anomalies', zorder=5)

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Anomaly Detection in Time Series')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    
    @staticmethod
    def plot_training_data():
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load the CSV
        df = pd.read_csv('../data/WeatherData.csv', parse_dates=['DATE'])

        # Convert the DATE column to datetime and set as index (optional)
        df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m')
        df.set_index('DATE', inplace=True)

        # Convert numeric columns (some may still be object due to empty values)
        df = df.apply(pd.to_numeric, errors='coerce')

        # 1. Average Temperature Over Time
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=df.index, y='TAVG', marker='o')
        plt.title('Average Monthly Temperature')
        plt.ylabel('Temperature (°F)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. TAVG, TMAX, TMIN
        df[['TAVG', 'TMAX', 'TMIN']].plot(figsize=(12, 6), title='Temperature Over Time', marker='o')
        plt.tight_layout()
        plt.show()

        # 3. Precipitation and Snowfall
        df[['PRCP', 'SNOW']].fillna(0).plot.area(figsize=(12, 6), alpha=0.5)
        plt.title('Precipitation and Snowfall Over Time')
        plt.ylabel('Inches')
        plt.tight_layout()
        plt.show()

        # 4. Heatmap
        heatmap_data = df.copy()
        heatmap_data['Month'] = heatmap_data.index.month
        heatmap_data['Year'] = heatmap_data.index.year
        pivot = heatmap_data.pivot_table(index='Year', columns='Month', values='TAVG')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, cmap='coolwarm', annot=True)
        plt.title('Monthly Average Temperature Heatmap')
        plt.tight_layout()
        plt.show()

        # 5. Dual-axis plot
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax2 = ax1.twinx()
        df['TAVG'].plot(ax=ax1, color='red', label='Avg Temp', marker='o')
        df['PRCP'].plot(ax=ax2, color='blue', label='Precipitation', marker='x')
        ax1.set_ylabel('Temperature (°F)')
        ax2.set_ylabel('Precipitation (inches)')
        ax1.set_title('Temperature vs. Precipitation')
        ax1.grid(True)
        plt.tight_layout()
        plt.show()

        # 6. Monthly Extreme High Temperatures
        df['EMXT'].plot(kind='bar', figsize=(12,6), title='Monthly Extreme High Temperatures')
        plt.ylabel('Temperature (°F)')
        plt.tight_layout()
        plt.show()


            



if __name__ == '__main__':
    visualize = Visualizer()
    visualize.plot_training_data()
