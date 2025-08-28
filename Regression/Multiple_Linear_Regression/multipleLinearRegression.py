# Importing the Libraries

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("50_Startups.csv");


# dataset.drop("Administration", axis=1, inplace=True);

print(dataset.corr(numeric_only=True));

# plt.scatter(dataset["Marketing Spend"],dataset["Profit"],color="blue");
# plt.grid(True);
# plt.show();


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

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

# Encoding the Categorical Data

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("State Encoding", OneHotEncoder(), [-1])
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

print(test_output);
