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
plt.title("Salary VS Experience (Training Set)");
plt.xlabel("Years Of Experience");
plt.ylabel("Salary");
plt.grid(True);
plt.show();

# Visualising the Test Set Results

plt.scatter(x_test, y_test, color="red");
plt.plot(x_train, regressor.predict(x_train), color="green");
plt.title("Salary VS Experience (Test Set)");
plt.xlabel("Years Of Experience");
plt.ylabel("Salary");
plt.grid(True);
plt.show();

# Parameters / Methods to Check Suitability of SLR

print(dataset.corr(numeric_only=True)); 

from sklearn.metrics import r2_score;

print("\n",r2_score(y,regressor.predict(x))); # R² (Coefficient of Determination)

from sklearn.metrics import mean_squared_error;

mse = mean_squared_error(y_test,y_predict); # Mean Squared Error (MSE)

# print("\n",mse); 

rmse = np.sqrt(mse);

print("\n",rmse); # Root Mean Squared Error (MSE)

print(((rmse*100)/np.mean(y_train))); # Comparing RMSE to the Mean of Y-Train
