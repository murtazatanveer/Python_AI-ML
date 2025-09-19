# Importing the Libraries
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("Social_Network_Ads.csv");

# Data Cleaning 

# print(dataset.info());

# print(dataset.isnull().sum());

print(dataset.corr(numeric_only=True));

# Removing Outliars

def removing_outliers_IQR(data, cols):
    mask = pd.Series(True, index=data.index)  
    
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
    
        mask &= (data[c] >= lowerBound) & (data[c] <= upperBound)
    
    return data[mask]

print(len(dataset))
dataset = removing_outliers_IQR(dataset,['Age','EstimatedSalary']);
print(len(dataset))

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

# Feature Scaling

from sklearn.preprocessing import StandardScaler;

sc = StandardScaler();
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)
x = sc.transform(x); 

# Training the Decision Tree model on the Training set

from sklearn.tree import DecisionTreeClassifier;
classifier = DecisionTreeClassifier(criterion="gini" , random_state=0);
classifier.fit(x_train,y_train);

# Predicting a new result

print(classifier.predict(sc.transform([[30,87000]])));

# Predicting the Test set results

# y_pred = classifier.predict(x_test);

# print(np.concatenate((y_test.reshape(-1,1), y_pred.reshape(-1,1)), axis=1));

# Evaluating the Model Performance

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classification_suitability_matrices import classification_suitability_Parameters

classification_suitability_Parameters(x,y, x_train, y_train, x_test, y_test, classifier);