import unittest
from unittest.mock import patch
from src.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):

    @patch("matplotlib.pyplot.show")
    def test_plot_temperature_trend(self, mock_show):
        years = list(range(2000, 2010))
        actual = [0.1 * i for i in range(10)]
        predicted = [0.12 * i for i in range(10)]
        Visualizer.plot_temperature_trend(years, actual, predicted, "Test Plot")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_future_predictions(self, mock_show):
        years = list(range(2025, 2030))
        predictions = [0.3 + 0.1 * i for i in range(5)]
        Visualizer.plot_future_predictions(years, predictions, "Future Test")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_clustered_data(self, mock_show):
        data = [(1, 2), (2, 3), (3, 4), (4, 5)]
        labels = [0, 1, 0, 1]
        Visualizer.plot_clustered_data(data, labels, "Clusters")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_anomalies(self, mock_show):
        data = [1, 2, 3, 100, 5, 6]
        anomalies = [False, False, False, True, False, False]
        Visualizer.plot_anomalies(data, anomalies, "Anomaly Detection")
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_sarima_vs_actual(self, mock_show):
        years = list(range(2000, 2005))
        actual = [0.5, 0.6, 0.7, 0.8, 0.9]
        sarima = [0.52, 0.61, 0.68, 0.81, 0.91]
        Visualizer.plot_sarima_vs_actual(years, actual, sarima)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_plot_linear_vs_actual(self, mock_show):
        years = list(range(2010, 2015))
        actual = [1.0, 1.1, 1.2, 1.3, 1.4]
        linear = [0.95, 1.05, 1.15, 1.25, 1.35]
        Visualizer.plot_linear_vs_actual(years, actual, linear)
        mock_show.assert_called_once()

if __name__ == '__main__':
    unittest.main()
