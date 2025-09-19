# Importing the Libraries

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("heart.csv"); 

# Data Cleaning

# print(dataset.info());

# print(dataset.isnull().sum());

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
dataset = removing_outliers_IQR(dataset,['Age','RestingBP',]);
print(len(dataset))

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("Country Encoding", OneHotEncoder(drop='first'), [1,2,6,8,10])
    ],
    remainder="passthrough"
);

x = np.array(ct.fit_transform(x));

# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder;

le = LabelEncoder();
y=le.fit_transform(y);

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Feature Scaling

from sklearn.preprocessing import StandardScaler;

sc = StandardScaler();
x_train = sc.fit_transform(x_train) 
x_test = sc.transform(x_test)
x = sc.transform(x); 

# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression;

classifier =  LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
classifier.fit(x_train,y_train);

# Evaluating the Model Performance

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from classification_suitability_matrices import classification_suitability_Parameters

classification_suitability_Parameters(x,y, x_train, y_train, x_test, y_test, classifier);