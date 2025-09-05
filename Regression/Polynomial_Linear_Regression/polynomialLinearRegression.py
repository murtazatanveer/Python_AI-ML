# Importing the Libraries 

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.linear_model import LinearRegression;

# Importing the Dataset

dataset = pd.read_csv("Position_Salaries.csv");

x = dataset.iloc[:, 1:-1].values;
y = dataset["Salary"].values;

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

# Visualizing Cleaned Dataset

# plt.scatter(x,y, color="red");
# plt.title("Level VS Salary Dataset Visualization");
# plt.xlabel("Level");
# plt.ylabel("Salary");
# plt.grid(True);
# plt.show();

# Choosing the Right Degree

# from sklearn.model_selection import cross_val_score


# degrees = [1, 2, 3, 4, 5]
# results = {}

# for d in degrees:
#     poly = PolynomialFeatures(degree=d)
#     X_poly = poly.fit_transform(x)

#     model = LinearRegression()
#     scores = cross_val_score(model, X_poly, y, cv=5, scoring="r2")  # 5-Fold CV
#     results[d] = np.mean(scores)

# for d, score in results.items():
#     print(f"Degree {d}: Average R² = {score:.3f}")

# Training the Simple Linear Regression Model on the Whole DataSet

lin_reg = LinearRegression();
lin_reg.fit(x,y);

# Training the Polynomial Linear Regression Model on the Whole DataSet

poly_reg = PolynomialFeatures(degree=4);
x_poly = poly_reg.fit_transform(x_train);
lin_reg_2 = LinearRegression();
lin_reg_2.fit(x_poly,y_train);


# Visulizing the Linear Regression Results

# plt.scatter(x,y, color="red");
# plt.plot(x,lin_reg.predict(x));
# plt.title("Visulizing the Linear Regression Results");
# plt.xlabel("Level");
# plt.ylabel("Salary");
# plt.grid(True);
# plt.show();


# # Visulizing the Polynomial Regression Results on Training Set

# plt.scatter(x_train,y_train, color="red");
# plt.plot(x,lin_reg_2.predict(x_poly));
# plt.title("Visulizing the Polynomial Regression Results");
# plt.xlabel("Level");
# plt.ylabel("Salary");
# plt.grid(True);
# plt.show();

# Visualising the Polynomial Regression results on Training Set (for higher resolution and smoother curve)

X_grid = np.arange(x.min(), x.max(), 0.1).reshape(-1, 1)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True);
plt.show()

# Visualising the Polynomial Regression results on Test Set (for higher resolution and smoother curve)


plt.scatter(x_test, y_test, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.grid(True);
plt.show()

# Parameters / Methods to Check Suitability of MLR


x_predict = lin_reg_2.predict(poly_reg.fit_transform(x));

x_train_predict = lin_reg_2.predict(x_poly);

x_test_predict = lin_reg_2.predict(poly_reg.fit_transform(x_test));

from sklearn.metrics import r2_score;

r2 = r2_score(y,x_predict);

r2_train = r2_score(y_train,x_train_predict);

r2_test = r2_score(y_test,x_test_predict);

print("\nR² (Coefficient of Determination) X and Y : ",r2); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Train and Y_Train : ",r2_train); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Test and Y_Test : ",r2_test); # R² (Coefficient of Determination)

# print("\nAdjusted R² w.r.t X and Y : ",1 - (1 - r2) * (len(y)  - 1) / (len(y)  - x.shape[1] - 1)); # Adjusted  R² 

# print("\nAdjusted R² w.r.t X_Train and Y_Train : ",1 - (1 - r2_train) * (len(y_train)  - 1) / (len(y_train)  - x_train.shape[1] - 1)); # Adjusted  R² 

# print("\nAdjusted R² w.r.t X_Test and Y_Test : ",1 - (1 - r2_test) * (len(y_test)  - 1) / (len(y_test)  - x_test.shape[1] - 1)); # Adjusted  R² 


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

residuals = y - x_predict;

plt.scatter(x_predict, residuals);
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Homoscedasticity Check")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True);
plt.show()