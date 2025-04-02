#!/usr/bin/env python3
#
#  gui.py
#  
#  Copyright 2025 vishwas <vishwasks.reach@gmail.com>
#  
#  


import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
import random	
from tkinter import messagebox
import subprocess
import sys
import threading

def run_main():
    model = main_model_var.get()
    loading_label.grid(row=0, column=0, sticky="nsew", columnspan=1) #show loading
    threading.Thread(target=run_main_thread, args=(model,)).start()

def run_main_thread(model):
    try:
        command = ["python3", "main.py","-m", model]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        main_output_text.delete(1.0, tk.END)
        main_output_text.insert(tk.END, result.stdout)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred:\n{e.stderr}")
    except FileNotFoundError:
        messagebox.showerror("Error", "main.py not found or Python interpreter not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    loading_label.grid_forget() #hide loading

def run_data_analysis():
    loading_label.grid(row=0, column=0, sticky="nsew", columnspan=1) #show loading
    threading.Thread(target=run_data_analysis_thread).start()

def run_data_analysis_thread():
    try:
        command = [sys.executable, "data_analysis.py"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data_analysis_output_text.delete(1.0, tk.END)
        data_analysis_output_text.insert(tk.END, result.stdout)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred:\n{e.stderr}")
    except FileNotFoundError:
        messagebox.showerror("Error", "data_analysis.py not found or Python interpreter not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    loading_label.grid_forget() #hide loading

root = tk.Tk()
root.title("Model and Data Analysis")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

notebook = ttk.Notebook(root)
notebook.grid(row=0, column=0, sticky="nsew")

# Main Tab
main_tab = ttk.Frame(notebook)
notebook.add(main_tab, text="Model Training")

main_tab.columnconfigure(0, weight=1)
main_tab.rowconfigure(3, weight=1)

main_model_label = tk.Label(main_tab, text="Select Model:")
main_model_label.grid(row=0, column=0, sticky="w")

main_model_var = tk.StringVar(main_tab)
main_model_var.set("rf")

main_model_dropdown = tk.OptionMenu(main_tab, main_model_var, "rf", "lr", "both")
main_model_dropdown.grid(row=1, column=0, sticky="w")

main_run_button = tk.Button(main_tab, text="Test", command=run_main)
main_run_button.grid(row=2, column=0, sticky="w")

main_output_label = tk.Label(main_tab, text="Model Training Output:")
main_output_label.grid(row=3, column=0, sticky="nw")

main_output_text_frame = tk.Frame(main_tab)  # Frame to hold text and scrollbar
main_output_text_frame.grid(row=3, column=0, sticky="nsew")

main_output_text = tk.Text(main_output_text_frame, height=15, width=80, bg="black", fg="green") # increased height
main_output_text.grid(row=0, column=0, sticky="nsew")

main_scrollbar = tk.Scrollbar(main_output_text_frame, command=main_output_text.yview)
main_scrollbar.grid(row=0, column=1, sticky="ns")

main_output_text.config(yscrollcommand=main_scrollbar.set)

# Data Analysis Tab
data_analysis_tab = ttk.Frame(notebook)
notebook.add(data_analysis_tab, text="Data Analysis")

data_analysis_tab.columnconfigure(0, weight=1)
data_analysis_tab.rowconfigure(1, weight=1)

data_analysis_run_button = tk.Button(data_analysis_tab, text="Run Data Analysis", command=run_data_analysis)
data_analysis_run_button.grid(row=0, column=0, sticky="w")

data_analysis_output_label = tk.Label(data_analysis_tab, text="Data Analysis Output:")
data_analysis_output_label.grid(row=1, column=0, sticky="nw")

data_analysis_output_text_frame = tk.Frame(data_analysis_tab)
data_analysis_output_text_frame.grid(row=1, column=0, sticky="nsew")

data_analysis_output_text = tk.Text(data_analysis_output_text_frame, height=15, width=80, bg="black", fg="green") #increased height
data_analysis_output_text.grid(row=0, column=0, sticky="nsew")

data_analysis_scrollbar = tk.Scrollbar(data_analysis_output_text_frame, command=data_analysis_output_text.yview)
data_analysis_scrollbar.grid(row=0, column=1, sticky="ns")

data_analysis_output_text.config(yscrollcommand=data_analysis_scrollbar.set)

# Loading Label
loading_label = tk.Label(root, text="Loading...", font=("Helvetica", 16))

root.mainloop()
