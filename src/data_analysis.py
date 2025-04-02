#!/usr/bin/env python3
#
#  data_analysis.py
#
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#
#


import sys
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from utilities import get_project_directories


if __name__ == '__main__':
    # The data is assumed to be in the dataset folder
    dataset_dir, output_dir, data_tmp_dir = get_project_directories()
    df = pd.read_csv(dataset_dir+'/network-slicing.csv')
    pprint(df.head())
    unique_slice_types = df['Slice Type (Output)'].unique()
    print(unique_slice_types)
    # Data Shape
    print("Data Shape:", df.shape)

    # Data Types
    print("\nData Types:\n", df.dtypes)

    # Descriptive Statistics
    print("\nDescriptive Statistics:\n", df.describe())

    # Missing Values
    print("\nMissing Values:\n", df.isnull().sum())

    # Unique Values and Frequencies for Categorical Columns
    for column in df.select_dtypes(include=['object']).columns:
      print(f"\nUnique values and frequencies for '{column}':")
      print(df[column].value_counts())

    # Correlation Analysis (for numerical variables)
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    print("\nCorrelation Matrix:\n", correlation_matrix)

    print("Plotting Distribution of data....")
    numerical_columns = ['Time (Input 5)', 'QCI (Input 6)', 'Packet Loss Rate (Reliability)']
    plt.figure(figsize=(15, 5))

    for i, column in enumerate(numerical_columns):
      plt.subplot(1, 3, i + 1)
      plt.hist(df[column], bins=20)
      plt.title(f'Distribution of {column}')
      plt.xlabel(column)
      plt.ylabel('Frequency')
      plt.savefig(output_dir+"/Distribution of "+column+".png")

    print(f"Images saved....")

    print("Plotting Frequency of data....")
    categorical_columns = ['Use CaseType (Input 1)', 'Slice Type (Output)']
    plt.figure(figsize=(15, 5))

    for i, column in enumerate(categorical_columns):
      plt.subplot(1, 2, i + 1)
      df[column].value_counts().plot(kind='bar')
      plt.title(f'Frequency of {column}')
      plt.xlabel(column)
      plt.ylabel('Frequency')
      plt.xticks(rotation=45, ha='right')
      plt.savefig(output_dir+"/Frequency of "+column+".png")
      
    print(f"Images saved....")

