# Unit Testing for Data Processor

import unittest
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """
    Unit tests for the DataProcessor class.
    """

    def setUp(self):
        """
        Set up a sample dataset for testing.
        """
        self.processor = DataProcessor("test.csv")
        self.processor.data = pd.DataFrame({
            "DATE": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
            "TMAX": [50,55,None], # test that None gets dropped
            "TMIN": [30,35,40]
        })

    def test_clean_data(self):
        """
        Test if data is correctly cleaned and normalized.
        """
        cleaned_data = self.processor.clean_data()

        # Make sure missing values are dropped
        self.assertEqual(len(cleaned_data), 2)

        # Check that AVG_TEMP exists
        self.assertIn("AVG_TEMP", cleaned_data.columns)

        # Make sure values are between 0 and 1 (normalized)
        self.assertGreaterEqual(cleaned_data["AVG_TEMP"].min(), 0)
        self.assertLessEqual(cleaned_data["AVG_TEMP"].max(), 1)


    def test_get_features_and_target(self):
        """
        Test if features and target values are correct
        """
        self.processor.data["AVG_TEMP"] = [0.1,0.5,0.9]
        features, target = self.processor.get_features_and_targets()

        # Check number of rows and columns
        self.assertEqual(features.shape, (3,2))
        self.assertEqual(target.shape, (3,))

        # Check feature values
        self.assertTrue(np.array_equal(features[:, 0], [2023, 2023, 2023]))
        self.assertTrue(np.array_equal(features[:, 1], [1, 2, 3]))

if __name__ == '__main__':
    unittest.main()