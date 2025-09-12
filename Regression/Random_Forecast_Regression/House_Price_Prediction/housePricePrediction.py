# Importing the Libraries

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


# Importing the Dataset

dataset = pd.read_csv("house_prices_dataset.csv");

# Data Cleaning 

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns

# print(dataset.info());
# print(dataset.isnull().sum());

# Removing Outliers

# print("\n\nBefore Removing Outliars Size of Dataset is ",len(dataset));

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

# dataset = removing_outliers_IQR(dataset,['square_feet','num_rooms','age','distance_to_city(km)','price']);

# print("\n\nAfter Removing Outliars Size of Dataset is ",len(dataset));


# Visualization Of Dataset

def datasetVisualization(cols,y_axis,data):
        for c in cols:
            plt.scatter(data[c],data[y_axis]);
            plt.title("Data Visualization");
            plt.xlabel(c);
            plt.ylabel(y_axis);
            plt.grid(True);
            plt.show();

# datasetVisualization(['square_feet','num_rooms','age','distance_to_city(km)'],'price',dataset);

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1];
y = dataset.iloc[:,-1];

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Training the Random Forest Regression model on the Training dataset

from sklearn.ensemble import RandomForestRegressor;
regressor = RandomForestRegressor(n_estimators=500,random_state=42);
regressor.fit(x_train,y_train);


# # Predicting the Test Set Results

# np.set_printoptions(precision=2);

# y_predict = regressor.predict(x_test);

# test_output = np.concatenate((y_predict.reshape(-1,1), y_test.to_numpy().reshape(-1, 1)), axis=1);

# # print("\n\n",test_output);


# Parameters / Methods to Check Suitability of Random Forecast Regression

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

