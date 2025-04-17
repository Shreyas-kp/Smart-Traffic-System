import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class TrafficDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')
        
    def load_and_clean_data(self):
        """Load and clean all three datasets"""
        # Load all datasets
        traffic_data = pd.read_csv('traffic_data.csv')
        traffic_patterns = pd.read_csv('traffic_patterns.csv')
        vehicle_data = pd.read_csv('vehicle_data.csv')
        
        # Clean each dataset
        self.clean_traffic_data(traffic_data)
        self.clean_traffic_patterns(traffic_patterns)
        self.clean_vehicle_data(vehicle_data)
        
        return traffic_data, traffic_patterns, vehicle_data
    
    def clean_traffic_data(self, df):
        """Clean traffic_data.csv"""
        # Handle missing values
        df.fillna({
            'hour': df['hour'].median(),
            'day': df['day'].mode()[0],
            'weather': df['weather'].mode()[0],
            'vehicle_count': df['vehicle_count'].mean()
        }, inplace=True)
        
        # Ensure proper data types
        df['hour'] = df['hour'].astype(int)
        df['day'] = df['day'].astype(int)
        df['weather'] = df['weather'].astype(int)
        
        # Remove outliers in vehicle_count
        q1 = df['vehicle_count'].quantile(0.25)
        q3 = df['vehicle_count'].quantile(0.75)
        iqr = q3 - q1
        df = df[~((df['vehicle_count'] < (q1 - 1.5 * iqr)) | 
                 (df['vehicle_count'] > (q3 + 1.5 * iqr)))]
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/7)
        
        return df
    
    def clean_traffic_patterns(self, df):
        """Clean traffic_patterns.csv"""
        # Handle missing values
        df.fillna({
            'average_speed': df['average_speed'].mean(),
            'vehicle_density': df['vehicle_density'].median(),
            'average_waiting_time': df['average_waiting_time'].mean(),
            'flow_direction_ratio': df['flow_direction_ratio'].median()
        }, inplace=True)
        
        # Cap values at reasonable limits
        df['vehicle_density'] = df['vehicle_density'].clip(0, 1)
        df['flow_direction_ratio'] = df['flow_direction_ratio'].clip(0, 1)
        
        # Remove outliers
        for col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
        
        return df
    
    def clean_vehicle_data(self, df):
        """Clean vehicle_data.csv"""
        # Handle missing values
        df.fillna({
            'length': df['length'].median(),
            'width': df['width'].median(),
            'speed': df['speed'].median(),
            'shape_factor': df['shape_factor'].median(),
            'vehicle_type': df['vehicle_type'].mode()[0]
        }, inplace=True)
        
        # Remove impossible values
        df = df[(df['length'] > 0) & (df['width'] > 0) & 
               (df['speed'] > 0) & (df['speed'] < 150) & 
               (df['shape_factor'] > 0) & (df['shape_factor'] <= 1)]
        
        # Remove outliers
        for col in ['length', 'width', 'speed', 'shape_factor']:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[~((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr)))]
        
        return df
    
    def prepare_traffic_flow_data(self, df):
        """Prepare data for traffic flow prediction (Linear Regression)"""
        # Features and target
        X = df[['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'weather']]
        y = df['vehicle_count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_vehicle_classification_data(self, df):
        """Prepare data for vehicle classification (Decision Tree)"""
        # Features and target
        X = df[['length', 'width', 'speed', 'shape_factor']]
        y = df['vehicle_type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Scale features (optional for Decision Trees)
        X_train = self.minmax_scaler.fit_transform(X_train)
        X_test = self.minmax_scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def prepare_traffic_pattern_data(self, df):
        """Prepare data for traffic pattern analysis (K-Means)"""
        # Features only for clustering
        X = df[['average_speed', 'vehicle_density', 
               'average_waiting_time', 'flow_direction_ratio']]
        
        # Split data (optional for clustering)
        X_train, X_test = train_test_split(
            X, test_size=0.2, random_state=42)
        
        # Scale features (important for K-Means)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test