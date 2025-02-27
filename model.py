# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # Use joblib instead of pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    """Load the dataset from the given filepath."""
    try:
        data = pd.read_csv("E:\Python\Cloudburst Prediction System\Rainfall.csv")
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data: handle missing values, drop unnecessary columns, and downsample."""
    try:
        # Strip whitespace from column names
        data.columns = data.columns.str.strip()

        # Drop the 'day' column
        data = data.drop(columns=['day'])

        # Fill missing values with the mean
        data['winddirection'].fillna(data['winddirection'].mean(), inplace=True)
        data['windspeed'].fillna(data['windspeed'].mean(), inplace=True)

        # Map 'rainfall' column to binary values
        data['rainfall'] = data['rainfall'].map({"yes": 1, "no": 0})

        # Drop highly correlated columns
        data = data.drop(columns=['maxtemp', 'temparature', 'mintemp'])

        # Downsample the majority class
        df_majority = data[data['rainfall'] == 1]
        df_minority = data[data['rainfall'] == 0]
        df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
        df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

        logging.info("Data preprocessing completed.")
        return df_downsampled
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def train_model(x_train, y_train):
    """Train a RandomForestClassifier using GridSearchCV."""
    try:
        rf_model = RandomForestClassifier(random_state=42)
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_features': ["sqrt", "log2"],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
        grid_search_rf.fit(x_train, y_train)
        best_rf_model = grid_search_rf.best_estimator_
        logging.info("Model training completed.")
        return best_rf_model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def evaluate_model(model, x_test, y_test):
    """Evaluate the model's performance on the test set."""
    try:
        y_pred = model.predict(x_test)
        logging.info("Model evaluation results:")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

def save_model(model, feature_names, filepath):
    """Save the trained model and feature names using joblib."""
    try:
        model_data = {"model": model, "feature_names": feature_names}
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def main():
    """Main function to execute the script."""
    try:
        # Load data
        filepath = "E:\\Python\\Cloudburst Prediction System\\Rainfall.csv"
        data = load_data(filepath)

        # Preprocess data
        df_downsampled = preprocess_data(data)

        # Split data into features and target
        x = df_downsampled.drop(columns=['rainfall'])
        y = df_downsampled['rainfall']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train the model
        best_rf_model = train_model(x_train, y_train)

        # Evaluate the model
        evaluate_model(best_rf_model, x_test, y_test)

        # Save the model
        save_model(best_rf_model, x.columns.tolist(), "rainfall_prediction_model.joblib")
    except Exception as e:
        logging.error(f"Script failed: {e}")

if __name__ == "__main__":
    main()