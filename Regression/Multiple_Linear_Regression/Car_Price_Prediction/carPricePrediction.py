
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math;
import statsmodels.api as sm

# Importing the Dataset

dataset = pd.read_csv("CarPrice_Prediction.csv");

# Data Cleaning 

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns
