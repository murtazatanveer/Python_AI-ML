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
    for c in cols:
        Q1 = data[c].quantile(0.25)
        Q3 = data[c].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
       
        data = data[(data[c] >= lowerBound) & (data[c] <= upperBound)]
    
    return data

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
sc.fit_transform(x_train);
sc.transform(x_test)

# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression;

classifier = LogisticRegression(random_state=0);
classifier.fit(x_train,y_train);

# Predicting a new result

print(classifier.predict(sc.transform([[30,87000]])));

# Predicting the Test set results

# y_pred = classifier.predict(x_test);

# print(np.concatenate((y_test.reshape(-1,1), y_pred.reshape(-1,1)), axis=1));

# Methods / Parameters to check Suitability of Classification 

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, roc_curve
)

y_train_predict = classifier.predict(x_train)
y_predict = classifier.predict(x)
y_test_predict = classifier.predict(x_test);

y_prob = classifier.predict(x)[:, 1];
y_test_prob = classifier.predict(x_test)[:, 1];
y_train_prob = classifier.predict(x_train)[:, 1];

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

print("ROC-AUC (Whole Set) :", roc_auc_score(y, y_prob));
print("ROC-AUC (Training Set) :", roc_auc_score(y_train, y_train_prob));
print("ROC-AUC (Test Set) :", roc_auc_score(y_test, y_test_prob));

# 5. Log Loss

print("Log Loss (Whole Set) :", log_loss(y, y_prob));
print("Log Loss (Training Set) :", log_loss(y_train, y_train_prob));
print("Log Loss (Test Set) :", log_loss(y_test, y_test_prob));