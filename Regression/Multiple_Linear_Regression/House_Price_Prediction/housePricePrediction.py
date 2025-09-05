# Importing the Libraries

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import statsmodels.api as sm;


# Importing the Dataset

dataset = pd.read_csv("House Price Prediction Dataset.csv");

# Data Cleaning 

dataset.drop(dataset.columns[[0]], axis=1, inplace=True);

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns

# print(dataset.info());

def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data.loc[data[c] < lowerBound, c] = int(lowerBound)
        data.loc[data[c] > upperBound, c] = int(upperBound)
    
    return data

# print(len(dataset))
# dataset = removing_outliers_IQR(dataset,['Area', 'Bedrooms', 'Bathrooms', 'Floors' ,'YearBuilt','Price']);
# print(len(dataset))

# Visualization Of Dataset

def datasetVisualization(cols,y_axis,data):
        for c in cols:
            plt.scatter(data[c],data[y_axis]);
            plt.title("Data Visualization");
            plt.xlabel(c);
            plt.ylabel(y_axis);
            plt.grid(True);
            plt.show();

# datasetVisualization(['Area', 'Bedrooms', 'Bathrooms', 'Floors' ,'YearBuilt'],"Price",dataset);


# Ways to Check Multicollinearity Before Modal Training

x_p = dataset.drop("Price", axis=1)
x_p = pd.get_dummies(x_p, drop_first=True, dtype=int)
y_p = dataset["Price"]

x_p = x_p.drop(["YearBuilt"], axis=1);

print(dataset.corr(numeric_only=True),"\n");

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---- VIF ----

vif_data = pd.DataFrame()
vif_data["Feature"] = x_p.columns
vif_data["VIF"] = [variance_inflation_factor(x_p.values, i) for i in range(x_p.shape[1])]
print("\nVariance Inflation Factor (VIF):\n", vif_data)

# ---- Tolerance ----
tolerance_data = []
for feature, vif in zip(x_p.columns, vif_data["VIF"]):
    tolerance_data.append((feature, 1/vif))

print("\nTolerance values:")
for feature, tol in tolerance_data:
    print(f"{feature}: {tol}")

# ---- Condition Number ----
correlationMatrix = x_p.corr().values
cond_number = np.linalg.cond(correlationMatrix)
print("\nCondition Number:", cond_number)


# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = x_p.values;
y = dataset.iloc[:,-1].values;


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

# Parameters / Methods to Check Suitability of MLR

x_predict = regressor.predict(x);

x_train_predict = regressor.predict(x_train);

x_test_predict = regressor.predict(x_test);

from sklearn.metrics import r2_score;

r2 = r2_score(y,x_predict);

r2_train = r2_score(y_train,x_train_predict);

r2_test = r2_score(y_test,x_test_predict);

print("\nR² (Coefficient of Determination) X and Y : ",r2); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Train and Y_Train : ",r2_train); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Test and Y_Test : ",r2_test); # R² (Coefficient of Determination)


print("\nAdjusted R² w.r.t X and Y : ",1 - (1 - r2) * (len(y)  - 1) / (len(y)  - x.shape[1] - 1)); # Adjusted  R² 

print("\nAdjusted R² w.r.t X_Train and Y_Train : ",1 - (1 - r2_train) * (len(y_train)  - 1) / (len(y_train)  - x_train.shape[1] - 1)); # Adjusted  R² 

print("\nAdjusted R² w.r.t X_Test and Y_Test : ",1 - (1 - r2_test) * (len(y_test)  - 1) / (len(y_test)  - x_test.shape[1] - 1)); # Adjusted  R² 


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

print("\nResult : This Dataset and Modal is Not Fit for Multiple Linear Regression\n Underfitted Modal and High Bias");