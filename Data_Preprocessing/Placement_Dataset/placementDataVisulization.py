import matplotlib.pyplot as plt;
import pandas as pd;
import numpy as np;


data = pd.read_csv("Placement_Dataset.csv");

data = data.sample(n=100, random_state=42); 

print(data.corr(numeric_only=True));

x = data.loc[:,"CGPA"].values;
y = data.loc[:,"Salary (INR LPA)"].values;

plt.scatter(x, y, color = "green");
plt.title("Cgpa Vs Salary Data Visulization");
plt.xlabel("CGPA");
plt.ylabel("Salary");
plt.grid(True);
plt.show();


