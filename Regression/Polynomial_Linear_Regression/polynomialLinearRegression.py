# Importing the Libraries 

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("Position_Salaries.csv");

x = dataset.iloc[:, 1:-1].values;
y = dataset["Salary"].values;

# Visualizing Cleaned Dataset

# plt.scatter(x,y, color="red");
# plt.title("Level VS Salary Dataset Visualization");
# plt.xlabel("Level");
# plt.ylabel("Salary");
# plt.grid(True);
# plt.show();

# Training the Simple Linear Regression Model on the Whole DataSet

from sklearn.linear_model import LinearRegression;

lin_reg = LinearRegression();
lin_reg.fit(x,y);

# Training the Polynomial Linear Regression Model on the Whole DataSet

from sklearn.preprocessing import PolynomialFeatures;

poly_reg = PolynomialFeatures(degree=2);
x_poly = poly_reg.fit_transform(x);

lin_reg_2 = LinearRegression();
lin_reg_2.fit(x_poly,y);

# Visulizing the Linear Regression Results

plt.scatter(x,y, color="red");
plt.plot(x,lin_reg.predict(x));
plt.title("Visulizing the Linear Regression Results");
plt.xlabel("Level");
plt.ylabel("Salary");
plt.grid(True);
plt.show();

# Visulizing the Polynomial Regression Results

plt.scatter(x,y, color="red");
plt.plot(x,lin_reg_2.predict(x_poly));
plt.title("Visulizing the Linear Regression Results");
plt.xlabel("Level");
plt.ylabel("Salary");
plt.grid(True);
plt.show();
