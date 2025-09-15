# Importing the Libraries

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("CollegePlacement.csv");

# Data Cleaning

dataset.drop(columns=['College_ID', 'Prev_Sem_Result'], inplace=True);

# print(dataset.info());

# print(dataset.isnull().sum());

# print(dataset.corr(numeric_only=True));


# Removing Outliars

def removing_outliers_IQR(data, cols):
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data

print(len(dataset))
dataset = removing_outliers_IQR(dataset,['IQ','CGPA','Academic_Performance','Extra_Curricular_Score','Communication_Skills','Projects_Completed']);
print(len(dataset)) # LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

# Feature Scaling

from sklearn.preprocessing import StandardScaler;

sc = StandardScaler();
sc.fit_transform(x_train);
sc.transform(x_test)