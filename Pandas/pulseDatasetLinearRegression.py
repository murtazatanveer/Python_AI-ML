import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

pulse = pd.read_csv("pulse.csv");

pulse = pulse.iloc[:,[2,4]];

print("Correlation : \n",pulse.corr(numeric_only=True));

# Total Missing Values In Each Column

print("\n\nTotal Missing Values In Each Column\n",pulse.isnull().sum());


# Taking Care of Missing Data

from sklearn.impute import SimpleImputer;

missingValue = SimpleImputer(missing_values=np.nan, strategy="mean")
pulse[["Calories"]] = missingValue.fit_transform(pulse[["Calories"]]);

# Removing Outliers

print("\n\nBefore Removing Outliars Size of Dataset is ",len(pulse));

def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data


pulse = removing_outliers_IQR(pulse,["Pulse","Calories"]);

print("\n\nAfter Removing Outliars Size of Dataset is ",len(pulse));

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = pulse.iloc[:,0].values; 

y = pulse.iloc[:,1].values;


# Visualizing Cleaned Dataset

plt.scatter(x,y, color="red");
plt.title("Pulse VS Calories Data Visualization");
plt.xlabel("Pulse");
plt.ylabel("Calories");
plt.grid(True);
plt.show();

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x = x.reshape(-1, 1);
y = y.reshape(-1, 1);

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Training the Simple Linear Regression Model on the Training Set

from sklearn.linear_model import LinearRegression;

regressor = LinearRegression();
regressor.fit(x_train,y_train);

y_predict = regressor.predict(x_test);

 # Visualising the Training Set Results

plt.scatter(x_train, y_train, color="red");
plt.plot(x_train, regressor.predict(x_train), color="green");
plt.title("Pulse VS Calories (Training Set)");
plt.xlabel("Pulse");
plt.ylabel("Calories");
plt.grid(True);
plt.show();

# Visualising the Test Set Results

plt.scatter(x_test, y_test, color="red");
plt.plot(x_train, regressor.predict(x_train), color="green");
plt.title("Pulse VS Calories (Test Set)");
plt.xlabel("Pulse");
plt.ylabel("Calories");
plt.grid(True);
plt.show();


# Parameters / Methods to Check Suitability of SLR

print("\nCorrelation : \n",pulse.corr(),"\n");

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


print("Result : This Dataset and Modal is Fit for Simple Linear Regression")