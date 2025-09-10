# Importing the Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

# Importing the Dataset


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

print("After Removing Outliars Size of Dataset is ",len(pulse));

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = pulse.iloc[:,0].values; 
y = pulse.iloc[:,1].values;
x = x.reshape(-1, 1);

print("\nMin Indepdent Variable Value : ",min(x));
print("Max Indepdent Variable Value : ",max(x));

# Visualizing Cleaned Dataset

plt.scatter(x,y, color="red");
plt.title("Pulse VS Calories Data Visualization");
plt.xlabel("Pulse");
plt.ylabel("Calories");
plt.grid(True);
plt.show();

# Training the Decision Tree Regression model on the whole dataset

from sklearn.tree import DecisionTreeRegressor;

regressor = DecisionTreeRegressor(random_state=0);
regressor.fit(x,y);

# Predicting a New Result;

print("\nPredicting Result : ",regressor.predict([[105.5]]));

# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(np.min(x), np.max(x), 0.05);
X_grid = X_grid.reshape((len(X_grid), 1));
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Parameters / Methods to Check Suitability of MLR

x_predict = regressor.predict(x);


from sklearn.metrics import r2_score;

r2 = r2_score(y,x_predict);


print("\nR² (Coefficient of Determination) X and Y : ",r2); # R² (Coefficient of Determination)

print("\nAdjusted R² w.r.t X and Y : ",1 - (1 - r2) * (len(y)  - 1) / (len(y)  - x.shape[1] - 1)); # Adjusted  R² 

from sklearn.metrics import mean_squared_error;

mse_x = mean_squared_error(y,x_predict); # Mean Squared Error (MSE)

rmse_x = np.sqrt(mse_x);

print("\nRoot Mean Squared Error (MSE) X and Y : ",rmse_x); # Root Mean Squared Error (MSE)

print("\nComparing RMSE to the Mean of Y in % : ",((rmse_x*100)/np.mean(y))); # Comparing RMSE to the Mean

print("\nMean Absolute Error X and Y: ",np.mean(np.mean(np.abs(x_predict-y))));

