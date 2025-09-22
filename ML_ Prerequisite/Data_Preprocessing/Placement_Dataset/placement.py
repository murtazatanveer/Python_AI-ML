# Importing Libraries

import pandas as pd;
import numpy as np;
from sklearn import preprocessing;

# Importing the Dataset

dataset = pd.read_csv("Placement_Dataset.csv" , index_col="Student_ID");

# print(dataset.head(10));

# Datatype Conversion and Data Cleaning

dataset.drop("Salary (INR LPA)", axis=1, inplace=True);

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns

# dataset.info();

# Total Missing Value in Each Column


# print(dataset.isnull().sum()); # No Missing Value is found in the Dataset

# Handling Duplicated;

# print(dataset.duplicated());
dataset.drop_duplicates(inplace=True);
print(len(dataset));

# Removing Outliers

def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data

dataset = removing_outliers_IQR(dataset,["CGPA","Internships"]);

print(len(dataset),"\n\n");

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;


# Encoding the Dependent Variable

le = preprocessing.LabelEncoder();
y = le.fit_transform(y);

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

# Feature Scaling 

scaler = preprocessing.StandardScaler();
x_train = scaler.fit_transform(x_train);
x_test = scaler.transform(x_test);

print(x_train[0:10,:] , "\n\n" , x_test[0:10,:]);

print("\n",len(x_train),"\n",len(x_test));