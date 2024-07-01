import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Custom modules for logging and exception handling
from src.exception import CustomException
from src.logger import logging

# Data transformation and model trainer components
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    """
    Class for handling the data ingestion process.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to initiate the data ingestion process.
        """
        # Logging is essential for tracing and debugging errors.
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Creating directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving raw data to specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Use logging for tracing and debugging errors.
            logging.info("Train test split initiated")
            
            # Splitting the dataset into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving train and test sets to specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Handling exceptions and logging them
            raise CustomException(e, sys)

# Main Execution Block:
if __name__ == "__main__":
    # Creating an instance of DataIngestion and initiating data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Creating an instance of DataTransformation and initiating data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Creating an instance of ModelTrainer and initiating model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
