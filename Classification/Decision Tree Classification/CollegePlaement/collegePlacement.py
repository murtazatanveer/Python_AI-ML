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
        ("Country Encoding", OneHotEncoder(drop='first'), [3])
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

from sklearn.model_selection import cross_val_score
scores_f1 = cross_val_score(classifier, x, y, cv=5, scoring='f1')
print("F1 Scores:", scores_f1)

# Methods / Parameters to check Suitability of Classification 

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve
)

y_train_predict = classifier.predict(x_train)
y_predict = classifier.predict(x)
y_test_predict = classifier.predict(x_test);

y_prob = classifier.predict_proba(x)[:, 1];
y_test_prob = classifier.predict_proba(x_test)[:, 1];
y_train_prob = classifier.predict_proba(x_train)[:, 1];

# 1. Confusion Matrix

cm = confusion_matrix(y, y_predict);
cm_train = confusion_matrix(y_train, y_train_predict);
cm_test = confusion_matrix(y_test, y_test_predict);

print("\n\nConfusion Matrix (Whole Set) : \n", cm);
print("Confusion Matrix (Training Set) : \n", cm_train);
print("Confusion Matrix (Test Set) : \n", cm_test);

# 2. Accuracy

print("\nAccuracy (Whole Set) : ", accuracy_score(y, y_predict));
print("Accuracy (Training Set) : ", accuracy_score(y_train, y_train_predict));
print("Accuracy (Test Set) : ", accuracy_score(y_test, y_test_predict));

# 3. Precision, Recall, F1

print("\nPrecision (Whole Set) : ", precision_score(y, y_predict));
print("Precision (Training Set) : ", precision_score(y_train, y_train_predict));
print("Precision (Test Set) : ", precision_score(y_test, y_test_predict));

print("\nRecall (Whole Set) : ", recall_score(y, y_predict));
print("Recall (Training Set) : ", recall_score(y_train, y_train_predict));
print("Recall (Test Set) : ", recall_score(y_test, y_test_predict));

print("\nF1 Score (Whole Set) : ", f1_score(y, y_predict));
print("F1 Score (Training Set) : ", f1_score(y_train, y_train_predict));
print("F1 Score (Test Set) : ", f1_score(y_test, y_test_predict));

# 4. ROC-AUC

print("\nROC-AUC (Whole Set) :", roc_auc_score(y, y_prob));
print("ROC-AUC (Training Set) :", roc_auc_score(y_train, y_train_prob));
print("ROC-AUC (Test Set) :", roc_auc_score(y_test, y_test_prob));

# 5. Log Loss

print("\nLog Loss (Whole Set) :", log_loss(y, y_prob));
print("Log Loss (Training Set) :", log_loss(y_train, y_train_prob));
print("Log Loss (Test Set) :", log_loss(y_test, y_test_prob));