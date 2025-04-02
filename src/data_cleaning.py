#!/usr/bin/env python3
#
#  data_cleaning.py
#  
#  


import sys
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

from utilities import get_project_directories, handle_outliers_iqr

def main(args):
    return 0


if __name__ == '__main__':
    print("Identifying outliers....")
    # The data is assumed to be in the dataset folder
    dataset_dir, output_dir, data_tmp_dir = get_project_directories()
    df = pd.read_csv(dataset_dir+'/network-slicing.csv')
    
    numerical_columns = ['Time (Input 5)', 'QCI (Input 6)', 'Packet Loss Rate (Reliability)']
    plt.figure(figsize=(15, 5))

    for i, column in enumerate(numerical_columns):
      plt.subplot(1, 3, i + 1)
      plt.boxplot(df[column])
      plt.title(f'Box Plot of {column}')
      plt.ylabel(column)
      plt.savefig(output_dir+"/Box Plot of "+column+".png")
      
    for column in ['QCI (Input 6)', 'Packet Loss Rate (Reliability)']:
        df = handle_outliers_iqr(df, column)
    
    # Check for duplicate rows
    duplicate_rows = df[df.duplicated()]

    if not duplicate_rows.empty:
      print("Duplicate rows found:")
      pprint(duplicate_rows)
      # Remove duplicate rows
      df.drop_duplicates(inplace=True)
      print("Duplicate rows removed.")
    else:
      print("No duplicate rows found.")
      
    cleaned_file = data_tmp_dir + "/cleaned_data.csv"
    print(f"Saving cleaned data to csv: {cleaned_file}")
    df.to_csv(cleaned_file, index=False)
