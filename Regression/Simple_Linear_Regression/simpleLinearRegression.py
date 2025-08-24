# Importing the Libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("Salary_Data.csv");

x = dataset.iloc[:, :-1].values;
y = dataset["Salary"].values;

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Training the Simple Linear Regression Model on the Training Set

from sklearn.linear_model import LinearRegression;

regressor = LinearRegression();
regressor.fit(x_train,y_train);

# Predicting the Test Set Results

y_predict = regressor.predict(x_test);


# Visualising the Training Set Results

plt.scatter(x_train, y_train, color="red");
plt.plot(x_train, regressor.predict(x_train), color="green");
plt.xlabel("Years Of Experience");
plt.ylabel("Salary");
plt.grid(True);
plt.show();

# Visualising the Test Set Results

plt.scatter(x_test, y_test, color="red");
plt.plot(x_test, regressor.predict(x_test), color="green");
plt.xlabel("Years Of Experience");
plt.ylabel("Salary");
plt.grid(True);
plt.show();