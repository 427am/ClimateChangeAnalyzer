import csv
import pandas as pd
from typing import Tuple
import numpy as np


class DataProcessor:
    """
    A class to process climate data from a CSV file.
    This class loads, cleans, normalizes, and extracts features and targets from a csv file. 
    """
    def __init__(self, file_path: str):
        """
        Initialize file path and class variables.

        :param file_path: Path to CSV file with climate data.
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame: 
        """
        Load climate data and remove unnecessary columns
        
        :return: DataFrame with data, max temperature, and min temperature.
        """
        filtered_data = []
        
        # Read CSV file and extract necessary columns
        with open(self.file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                filtered_row = {
                    'DATE': pd.to_datetime(row['DATE']), 
                    'TMAX': row['TMAX'],
                    'TMIN': row['TMIN']
                }
                filtered_data.append(filtered_row)
        
        # Create DataFrame
        self.data = pd.DataFrame(filtered_data)
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset and normalize temperatures.

        - Converts temp values to numeric format.
        - Removes rows with missing values.
        - Computes and normalizes the avg temp.

        :return: cleaned and normalized DataFrame.
        """
        # Convert temperatures to numbers and errors become NaN
        self.data['TMAX'] = pd.to_numeric(self.data['TMAX'], errors='coerce')
        self.data['TMIN'] = pd.to_numeric(self.data['TMIN'], errors='coerce')

        # Drop all NaN values
        self.data.dropna(inplace=True)
        
        # Compute the average temperature
        self.data['AVG_TEMP'] = (self.data['TMAX'] + self.data['TMIN']) / 2
        
        # Compute min/max values for normalization
        avg_temp_min, avg_temp_max = self.data['AVG_TEMP'].min(), self.data['AVG_TEMP'].max()
        
        # Normalize the average temperature
        self.data['AVG_TEMP'] = (self.data['AVG_TEMP'] - avg_temp_min) / (avg_temp_max - avg_temp_min)
        
        return self.data
    
    def get_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target variable from the dataset.

        - Features: Year and Month.
        - Target: Avg temp.

        :return: Tuple containing feature matrix and target vector. 
        """
        # Change DATE to datetime format
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        
        # Features: Year and Month
        self.data['Year'] = self.data['DATE'].dt.year
        self.data['Month'] = self.data['DATE'].dt.month
        features = self.data[['Year', 'Month']].to_numpy()

        # Target is now the average temperature
        target = self.data['AVG_TEMP'].to_numpy()  
        
        return features, target
