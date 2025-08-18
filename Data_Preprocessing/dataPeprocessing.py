
# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the Dataset

dataset = pd.read_csv("Data.csv");

x = dataset.iloc[:, :-1].values;
y = dataset["Purchased"].values;

print(x)
print("\n",y);

