# Data Collection and Preprocessing

# IMPORTS
import csv
import pandas as pd
from typing import Tuple
import numpy as np


class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame: 
        """Load climate data and remove unnecessary columns"""
        filtered_data = []
        
        with open(self.file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                filtered_row = {
                    'DATE': row['DATE'],
                    'PRCP': row['PRCP'],
                    'TMAX': row['TMAX'],
                    'TMIN': row['TMIN']
                }
                filtered_data.append(filtered_row)
        self.data = pd.DataFrame(filtered_data)
        return self.data
    
    def clean_data(self) -> pd.DataFrame:
        """Remove rows with missing values"""
        self.data.dropna(inplace=True)
        
        self.data['TMAX'] = pd.to_numeric(self.data['TMAX'], errors='coerce')
        self.data['TMIN'] = pd.to_numeric(self.data['TMIN'], errors='coerce')
        
        max_temp = self.data['TMAX'].max()
        min_temp = self.data['TMIN'].min()
        
        self.data['TMAX'] = (self.data['TMAX'] - min_temp) / (max_temp - min_temp)
        self.data['TMIN'] = (self.data['TMIN'] - min_temp) / (max_temp - min_temp)
        
        return self.data
    
    def get_features_and_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into features (year, month) and targets (max temp)"""
        self.data['DATE'] = pd.to_datetime(self.data['DATE'])
        self.data['Year'] = self.data['DATE'].dt.year
        self.data['Month'] = self.data['DATE'].dt.month
        
        features = self.data[['Year', 'Month']].to_numpy()
        target = self.data['TMAX'].to_numpy()
        
        return features, target
    
    

