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
dataset = removing_outliers_IQR(dataset,['IQ','CGPA','Academic_Performance','Extra_Curricular_Score','Communication_Skills','Projects_Completed']);
print(len(dataset)) 

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)


x = dataset.iloc[:,:-1].values;

y = dataset.iloc[:,-1].values;

# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("Country Encoding", OneHotEncoder(), [3])
    ],
    remainder="passthrough"
);

x = np.array(ct.fit_transform(x));

noise = np.random.normal(0, 0.5, size=x.shape)
x = x + noise

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

from sklearn.tree import DecisionTreeClassifier;
classifier = DecisionTreeClassifier(criterion="gini" , random_state=0 , max_depth=4);
classifier.fit(x_train,y_train);



# Evaluating the Model Performance

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from classification_suitability_matrices import classification_suitability_Parameters

classification_suitability_Parameters(x,y, x_train, y_train, x_test, y_test, classifier);