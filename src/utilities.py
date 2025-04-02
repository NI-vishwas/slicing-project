#!/usr/bin/env python3
#
#  utilities.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#  
#  


import sys
from pathlib import Path
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def get_project_directories():
    """
    Determines the project's output and temporary data directories as strings.

    Assumes the project root is two levels above the current file's directory.

    Returns:
        tuple: A tuple containing the output directory and temporary data directory as strings.
    """
    current_file_path = Path(__file__).resolve()
    project_dir = current_file_path.parent.parent

    output_dir = project_dir / 'output'
    data_tmp_dir = project_dir / 'tmp'
    dataset_dir = project_dir / 'dataset'

    return str(dataset_dir),str(output_dir), str(data_tmp_dir)

def handle_outliers_iqr(df, column):
  """Handles outliers using the IQR method.

  Args:
    df: The DataFrame.
    column: The column name.

  Returns:
    The DataFrame with outliers handled.
  """
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_limit = Q1 - 1.5 * IQR
  upper_limit = Q3 + 1.5 * IQR
  df[column] = df[column].clip(lower_limit, upper_limit)
  return df
  
def prepare_data(X, y):
    """
    Prepares the data for machine learning, including one-hot encoding,
    missing value handling, data splitting, and feature scaling.

    Args:
        X (pd.DataFrame): The feature DataFrame.
        y (pd.Series or pd.DataFrame): The target Series or DataFrame.

    Returns:
        tuple: A tuple containing X_train, X_test, y_train, y_test.
    """

    # 1. One-hot encoding for categorical features, including the target variable
    categorical_features = ['Use CaseType (Input 1)', 'Technology Supported (Input 3)', 'Slice Type (Output)']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    all_categorical_data = pd.concat([X[categorical_features[:2]], y], axis=1)
    encoded_features = encoder.fit_transform(all_categorical_data)
    encoded_feature_names = encoder.get_feature_names_out(all_categorical_data.columns)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

    X_encoded = encoded_df[encoded_feature_names[:-3]]
    y_encoded = encoded_df[encoded_feature_names[-3:]]

    X = pd.concat([X[['QCI (Input 6)', 'Packet Loss Rate (Reliability)']], X_encoded], axis=1)

    # 2. Handle Missing Values in Target Variable
    y_encoded = y_encoded.fillna(y.mode().iloc[0])

    # 3. Handle Missing Values in Features and Encode Categorical Variables
    X = X.dropna()
    y_encoded = y_encoded.loc[X.index] #Align target variable with feature data after dropping rows

    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 5. Feature Scaling for numerical features
    numerical_features = ['QCI (Input 6)', 'Packet Loss Rate (Reliability)']
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # 6. Feature Scaling (Redundant, but included for consistency with original code)
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test, y_train, y_test

def train_evaluate_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest Classifier, makes predictions, and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame or pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame or pd.Series): Testing target.

    Returns:
        tuple: A tuple containing the Random Forest model, predictions, and evaluation metrics
               (accuracy, precision, recall, f1-score).
    """

    # Instantiate models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train models
    rf_model.fit(X_train, y_train)

    # Make predictions
    rf_predictions = rf_model.predict(X_test)

    # Evaluate Random Forest model
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    rf_precision = precision_score(y_test, rf_predictions, average='weighted')
    rf_recall = recall_score(y_test, rf_predictions, average='weighted')
    rf_f1 = f1_score(y_test, rf_predictions, average='weighted')

    print("Random Forest Model Performance:")
    print("Accuracy:", rf_accuracy)
    print("Precision:", rf_precision)
    print("Recall:", rf_recall)
    print("F1-score:", rf_f1)

    return rf_model, rf_predictions, (rf_accuracy, rf_precision, rf_recall, rf_f1)

def train_evaluate_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Trains a One-vs-Rest Logistic Regression model, makes predictions, and evaluates its performance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame or pd.Series): Training target.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.DataFrame or pd.Series): Testing target.

    Returns:
        tuple: A tuple containing the Logistic Regression model, predictions, and evaluation metrics
               (accuracy, precision, recall, f1-score).
    """

    lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # Evaluate Logistic Regression model
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_precision = precision_score(y_test, lr_predictions, average='weighted')
    lr_recall = recall_score(y_test, lr_predictions, average='weighted')
    lr_f1 = f1_score(y_test, lr_predictions, average='weighted')

    print("\nLogistic Regression Model Performance:")
    print("Accuracy:", lr_accuracy)
    print("Precision:", lr_precision)
    print("Recall:", lr_recall)
    print("F1-score:", lr_f1)

    return lr_model, lr_predictions, (lr_accuracy, lr_precision, lr_recall, lr_f1)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
