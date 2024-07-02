import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    file_path (str): The path to the file where the object will be saved.
    obj (object): The object to be saved.

    Raises:
    CustomException: If an error occurs during the save operation.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open the file in write-binary mode and save the object
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple machine learning models using GridSearchCV and returns their performance.

    Parameters:
    X_train (np.ndarray): Training data features.
    y_train (np.ndarray): Training data labels.
    X_test (np.ndarray): Testing data features.
    y_test (np.ndarray): Testing data labels.
    models (dict): Dictionary of models to be evaluated.
    param (dict): Dictionary of parameters for GridSearchCV for each model.

    Returns:
    dict: A report of the R2 scores for the test set for each model.

    Raises:
    CustomException: If an error occurs during the model evaluation.
    """
    try:
        report = {}
        # Iterates over each model in the models dictionary: For each model, performs grid search cross-validation to find the best hyperparameters
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            # Perform grid search with cross-validation
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set model to the best found parameters and fit on training data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    Loads a Python object from a file using pickle.

    Parameters:
    file_path (str): The path to the file from which the object will be loaded.

    Returns:
    object: The loaded object.

    Raises:
    CustomException: If an error occurs during the load operation.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
