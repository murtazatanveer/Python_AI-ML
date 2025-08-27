import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

sales = pd.read_csv("Temperature_VS_IceCream.csv");

print("\nCorrelation : \n",sales.corr());
print("\n\n",sales.info());

# Total Missing Values In Each Column

print("\n\nTotal Missing Values In Each Column\n",sales.isnull().sum());

# Removing Outliers

print("\n\nBefore Removing Outliars Size of Dataset is ",len(sales));

def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data


sales = removing_outliers_IQR(sales,["Temperature","Ice Cream Profits"]);

print("\n\nAfter Removing Outliars Size of Dataset is ",len(sales));

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = sales.iloc[:,0].values; 

y = sales.iloc[:,1].values;

# Visualizing Cleaned Dataset

# plt.scatter(x,y, color="red");
# plt.title("Temperature VS  Ice-Cream Sales Visualization");
# plt.xlabel("Temperature");
# plt.ylabel("Sales");
# plt.grid(True);
# plt.show();

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
plt.title("Temperature VS  Ice-Cream Sales (Training Set)");
plt.xlabel("Temperature");
plt.ylabel("Sales");
plt.grid(True);
plt.show();

# Visualising the Test Set Results

plt.scatter(x_test, y_test, color="red");
plt.plot(x_train, regressor.predict(x_train), color="green");
plt.title("Temperature VS  Ice-Cream Sales (Test Set)");
plt.xlabel("Temperature");
plt.ylabel("Sales");
plt.grid(True);
plt.show();


# Parameters / Methods to Check Suitability of SLR

from sklearn.metrics import r2_score;

print("\nR² (Coefficient of Determination) : ",r2_score(y,regressor.predict(x))); # R² (Coefficient of Determination)

from sklearn.metrics import mean_squared_error;

mse = mean_squared_error(y,regressor.predict(x)); # Mean Squared Error (MSE)

rmse = np.sqrt(mse);

print("\nRoot Mean Squared Error (MSE) : ",rmse); # Root Mean Squared Error (MSE)

print("\nComparing RMSE to the Mean of Y-Train in % : ",((rmse*100)/np.mean(y))); # Comparing RMSE to the Mean of Y-Train

print("Result : This Dataset and Modal is Fit for Simple Linear Regression")