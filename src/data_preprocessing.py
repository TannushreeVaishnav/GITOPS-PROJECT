import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException

logger=get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path,noise_level=0.03):
        self.input_path=input_path
        self.output_path=output_path
        self.df=None
        self.features=None
        self.noise_level=noise_level
        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data Processing Initalized")

    def load_data(self):
        try:
            self.df=pd.read_csv(self.input_path)
            logger.info("Dta loaded successfully")
        except Exception as e:
            logger.error(f" Error while loading the data {e}")
            raise CustomException("Failed to load the data",e)
        
    def preprocess(self):
        try:
            self.df["Timestamp"]=pd.to_datetime(self.df["Timestamp"],errors='coerce')
            categorical_cols=['Operation_Mode','Efficiency_Status']
            for col in categorical_cols:
                self.df[col]=self.df[col].astype('category')
            self.df["Year"]=self.df["Timestamp"].dt.year
            self.df["Month"]=self.df["Timestamp"].dt.month
            self.df['Day']=self.df['Timestamp'].dt.day
            self.df['Hour']=self.df['Timestamp'].dt.hour
            #Removing columns
            self.df.drop(columns=["Timestamp","Machine_ID","Error_Rate_%","Production_Speed_units_per_hr"],inplace=True)

            #Cyclical encoding for Hour
            self.df["Hour_sin"] = np.sin(2 * np.pi * self.df["Hour"] / 24)
            self.df["Hour_cos"] = np.cos(2 * np.pi * self.df["Hour"] / 24)

            #Label Encoding
            columns_to_encode=["Efficiency_Status","Operation_Mode"]
            for col in columns_to_encode:
                le=LabelEncoder()
                self.df[col]=le.fit_transform(self.df[col])

            logger.info("All basic data preprocessing done...")
        
        except Exception as e:
            logger.error(f" Error while prerocessing the data {e}")
            raise CustomException("Failed to preprocessing the data",e)
        
    def split_and_scale_and_save(self):
        try:
            self.features = [
                'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
                'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
                'Quality_Control_Defect_Rate_%', 'Predictive_Maintenance_Score',
                'Year', 'Month', 'Day', 'Hour_sin', 'Hour_cos'
            ]
            X = self.df[self.features]
            y = self.df["Efficiency_Status"]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Add noise only to training set
            num_col = X_train.select_dtypes(include=[np.number]).columns
            noise = np.random.normal(0, self.noise_level, X_train[num_col].shape)
            X_train_noisy = X_train.copy()
            X_train_noisy[num_col] += noise

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_noisy)
            X_test_scaled = scaler.transform(X_test)

            # Save artifacts
            joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test),
                        os.path.join(self.output_path, "train_test_split.pkl"))
            joblib.dump(scaler, os.path.join(self.output_path, "scaler.pkl"))
            joblib.dump(self.features, os.path.join(self.output_path, "features.pkl"))

            logger.info("Data split, noise added, scaled, and saved successfully.")

        except Exception as e:
            logger.error(f"Error while splitting/scaling/saving data {e}")
            raise CustomException("Failed to split, scale, and save data", e)

        
    def run(self):
        self.load_data()
        self.preprocess()
        self.split_and_scale_and_save()
if __name__=="__main__":
    processor=DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    processor.run()




        





            
        






