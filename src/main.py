#!/usr/bin/env python3
#
#  main.py
#
#


import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from utilities import get_project_directories, prepare_data, train_evaluate_random_forest,train_evaluate_logistic_regression


def main(args):
    dataset_dir, output_dir, data_tmp_dir = get_project_directories()
    CLEANED_DATA_FILE = os.path.join(data_tmp_dir, 'cleaned_data.csv')

    features = ['Use CaseType (Input 1)', 'Technology Supported (Input 3)', 'QCI (Input 6)', 'Packet Loss Rate (Reliability)']
    target = 'Slice Type (Output)'

    try:
        if not os.path.exists(CLEANED_DATA_FILE):
            raise IOError(f"{CLEANED_DATA_FILE} does not exist")
        else:
            df = pd.read_csv(CLEANED_DATA_FILE)
            X = df[features]
            y = df[target]

    except IOError as e:
        print(f"{e}")
        return 1  # Exit with an error code

    X_train, X_test, y_train, y_test = prepare_data(X, y)

    if args.model == 'rf' or args.model == 'both':
        rf_model, rf_predictions, rf_metrics = train_evaluate_random_forest(X_train, y_train, X_test, y_test)
        rf_accuracy, rf_precision, rf_recall, rf_f1 = rf_metrics
        print(f"Accuracy from returned tuple (RF): {rf_accuracy}")
        print(f"Trained Random Forest Model: {rf_model}")

    if args.model == 'lr' or args.model == 'both':
        lr_model, lr_predictions, lr_metrics = train_evaluate_logistic_regression(X_train, y_train, X_test, y_test)
        lr_accuracy, lr_precision, lr_recall, lr_f1 = lr_metrics
        print(f"Accuracy from returned tuple (LR): {lr_accuracy}")
        print(f"Trained Logistic Regression Model: {lr_model}")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate machine learning models.')
    parser.add_argument("--model", "-m", choices=['rf', 'lr', 'both'], help='Specify the model to train (rf, lr, or both).')
    args = parser.parse_args()
    sys.exit(main(args))

