import matplotlib.pyplot as plt;
import pandas as pd;
import numpy as np;


data = pd.read_csv("Placement_Dataset.csv");

print(data.corr(numeric_only=True));

x = data.loc[:20,"CGPA"].values;
y = data.loc[:20,"Salary (INR LPA)"].values;

plt.plot(x, y, color = "green");
plt.title("Cgpa Vs Salary Data Visulization");
plt.xlabel("CGPA");
plt.ylabel("Salary");
plt.grid(True);
plt.show();


