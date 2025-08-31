# Importing the Libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("Salary_Data.csv");


x = dataset.iloc[:, :-1].values;
y = dataset["Salary"].values;

# plt.scatter(x, y, color="red");
# plt.title("Salary VS Experience (Training Set)");
# plt.xlabel("Years Of Experience");
# plt.ylabel("Salary");
# plt.grid(True);
# plt.show();


# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Training the Simple Linear Regression Model on the Training Set

from sklearn.linear_model import LinearRegression;

regressor = LinearRegression();
regressor.fit(x_train,y_train);


# Predicting the Test Set Results

y_predict = regressor.predict(x_test);

y_train_predict = regressor.predict(x_train);

#  Visualising the Training Set Results

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


# Parameters / Methods to Check Suitability of MLR

x_predict = regressor.predict(x);

x_train_predict = regressor.predict(x_train);

x_test_predict = regressor.predict(x_test);

from sklearn.metrics import r2_score;

print("\nR² (Coefficient of Determination) X and Y : ",r2_score(y,x_predict)); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Train and Y_Train : ",r2_score(y_train,x_train_predict)); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Test and Y_Test : ",r2_score(y_test,x_test_predict)); # R² (Coefficient of Determination)


from sklearn.metrics import mean_squared_error;

mse_x = mean_squared_error(y,x_predict); # Mean Squared Error (MSE)

rmse_x = np.sqrt(mse_x);

mse_x_test = mean_squared_error(y_test,x_test_predict);

rmse_x_test = np.sqrt(mse_x_test);

mse_x_train = mean_squared_error(y_train,x_train_predict);

rmse_x_train = np.sqrt(mse_x_train);


print("\nRoot Mean Squared Error (MSE) X and Y : ",rmse_x); # Root Mean Squared Error (MSE)

print("\nRoot Mean Squared Error (MSE)  X_Train and Y_Train : ",rmse_x_train); # Root Mean Squared Error (MSE)

print("\nRoot Mean Squared Error (MSE)  X_Test and Y_Test : ",rmse_x_test); # Root Mean Squared Error (MSE)


print("\nComparing RMSE to the Mean of Y in % : ",((rmse_x*100)/np.mean(y))); # Comparing RMSE to the Mean

print("\nComparing RMSE to the Mean of Y_Train in % : ",((rmse_x_train*100)/np.mean(y_train))); # Comparing RMSE to the Mean

print("\nComparing RMSE to the Mean of Y_Test in % : ",((rmse_x_test*100)/np.mean(y_test))); # Comparing RMSE to the Mean


print("\nMean Absolute Error X and Y: ",np.mean(np.mean(np.abs(x_predict-y))));

print("\nMean Absolute Error X_Train and Y_Train : ",np.mean(np.mean(np.abs(x_train_predict-y_train))));

print("\nMean Absolute Error X_Test and Y_Test: ",np.mean(np.mean(np.abs(x_test_predict-y_test))));


# Homoscedasticity Check

residuals = y - regressor.predict(x);

plt.scatter(regressor.predict(x), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Homoscedasticity Check")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True);
plt.show()

print("Result : This Dataset and Modal is Fit for Simple Linear Regression")

