# Importing the Libraries

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math;
import scipy.stats as st


# Importing the Dataset

dataset = pd.read_csv("50_Startups.csv");

# dataset.drop("Administration", axis=1, inplace=True);

# Removing Outliers

print("\n\nBefore Removing Outliars Size of Dataset is ",len(dataset));


def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data


dataset = removing_outliers_IQR(dataset, ["R&D Spend","Administration","Marketing Spend","Profit"]);

print("\n\nAfter Removing Outliars Size of Dataset is ",len(dataset));

# Visualization Of Dataset

def datasetVisualization(cols,y_axis,data):
        for c in cols:
            plt.scatter(data[c],data[y_axis]);
            plt.title("Data Visualization");
            plt.xlabel(c);
            plt.ylabel(y_axis);
            plt.grid(True);
            plt.show();

# datasetVisualization(["R&D Spend","Administration","Marketing Spend"],"Profit",dataset);

# Ways to Check Multicollinearity Before Modal Training

print(dataset.corr(numeric_only=True));


X = dataset[["R&D Spend", "Administration", "Marketing Spend"]]

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X = sm.add_constant(X);
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\n\nVariance Inflation Factor (VIF)\n",vif_data)


tolerance_data = []
for feature, vif in zip(X.columns, vif_data["VIF"]):
    tolerance_data.append((feature, 1/vif))

print("\n\nTolerance values:")
for feature, tol in tolerance_data:
    print(f"{feature}: {tol}")


correlationMatrix = dataset.iloc[:,:-1];
correlationMatrix = correlationMatrix.corr(numeric_only=True).values;

cond_number = np.linalg.cond(correlationMatrix);
print("\nCondition Number:", cond_number);

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

# Encoding the Categorical Data

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("State Encoding", OneHotEncoder(drop="first"), [-1])
    ],
    remainder="passthrough"
);

x = np.array(ct.fit_transform(x));


# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

# Training the Multiple Linear Regression Model on the Training Set

from sklearn.linear_model import LinearRegression;

regressor = LinearRegression();
regressor.fit(x_train,y_train);


# Predicting the Test Set Results

np.set_printoptions(precision=2);

y_predict = regressor.predict(x_test);

test_output = np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), axis=1);

print("\n\n",test_output);

# Finding P-Value of the dataset

y_pred = regressor.predict(x) # Predictions
residuals = y - y_pred # Residuals

temp=x.copy();

n = temp.shape[0]     # samples
k = temp.shape[1]     # predictors
dfree = n - k - 1  # degrees of freedom
print(temp);
MSE = (residuals**2).sum() / dfree # Mean Squared Error
X_design = np.append(np.ones((n, 1)), temp, axis=1).astype(float) # Add intercept column and ensure float dtype
var_b = MSE * np.linalg.inv(X_design.T @ X_design).diagonal() # Variance-Covariance matrix
SE_b = np.sqrt(var_b) # Standard errors
coeffs = np.append(regressor.intercept_, regressor.coef_) # Coefficients (intercept + slopes)
t_stats = coeffs / SE_b # t-statistics
p_values = [2 * (1 - st.t.cdf(abs(t), dfree)) for t in t_stats]
p_values = [float(p) for p in p_values[-3:]]
print("P-values:", p_values) # p-values (two-tailed test)

# Parameters / Methods to Check Suitability of MLR

x_predict = regressor.predict(x);

x_test_predict = regressor.predict(x_test);

from sklearn.metrics import r2_score;

print("\nR² (Coefficient of Determination) : ",r2_score(y,x_predict)); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination)  : ",r2_score(y_test,x_test_predict)); # R² (Coefficient of Determination)


from sklearn.metrics import mean_squared_error;

mse_x = mean_squared_error(y,x_predict); # Mean Squared Error (MSE)

rmse_x = np.sqrt(mse_x);

mse_x_test = mean_squared_error(y_test,x_test_predict);

rmse_x_test = np.sqrt(mse_x_test);

print("\nRoot Mean Squared Error (MSE) X and Y : ",rmse_x); # Root Mean Squared Error (MSE)

print("\nRoot Mean Squared Error (MSE)  X_Test and Y_Test : ",np.sqrt(mean_squared_error(y_test,x_test_predict))); # Root Mean Squared Error (MSE)

print("\nComparing RMSE to the Mean of Y in % : ",((rmse_x*100)/np.mean(y))); # Comparing RMSE to the Mean

print("\nComparing RMSE to the Mean of Y_Test in % : ",((rmse_x_test*100)/np.mean(y_test))); # Comparing RMSE to the Mean


print("\nMean Absolute Error X and Y: ",np.mean(np.mean(np.abs(x_predict-y))));

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

print("\nResult : This Dataset and Modal is Fit for Multiple Linear Regression");

