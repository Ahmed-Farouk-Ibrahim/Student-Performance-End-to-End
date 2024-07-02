import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        """
        Predicts outcomes based on the input features using a pre-trained model and preprocessor.
        
        Args:
        features (pd.DataFrame): Input features for prediction.
        
        Returns:
        np.ndarray: Predicted outcomes.
        
        Raises:
        CustomException: If an exception occurs during prediction.
        """
        try:
            # Paths to the pre-trained model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            # Debugging logs
            print("Before Loading")
            
            # Load the pre-trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Debugging logs
            print("After Loading")
            
            # Preprocess the input features and make predictions
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, 
                 lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        """
        Initialize the CustomData class with student data attributes.
        
        Args:
        gender (str): Gender of the student.
        race_ethnicity (str): Race or ethnicity of the student.
        parental_level_of_education (str): Parental level of education.
        lunch (str): Lunch type.
        test_preparation_course (str): Test preparation course status.
        reading_score (int): Reading score out of 100.
        writing_score (int): Writing score out of 100.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        """
        Converts the student data into a DataFrame format.
        
        Returns:
        pd.DataFrame: DataFrame containing the student data.
        
        Raises:
        CustomException: If an exception occurs during DataFrame creation.
        """
        try:
            # Create a dictionary with the student data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            
            # Convert the dictionary to a DataFrame and return it
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
