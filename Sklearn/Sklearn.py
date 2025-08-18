import pandas as pd;
from sklearn import preprocessing;
from sklearn.impute import SimpleImputer;

# Load the dataset

data = pd.read_csv("Data.csv");

# Fill the Missing Values using Simple Imputer

fillAge = SimpleImputer(strategy="mean")
data["Age"] = fillAge.fit_transform(data[["Age"]]);

fillSalary = SimpleImputer(strategy="mean");
data["Salary"]=fillSalary.fit_transform(data[["Salary"]]);

print(data)

# Standardize the 'Age' and 'Salary' columns

# scaler = preprocessing.StandardScaler();

# scaler.fit(data[["Age","Salary"]]);
# data[["Age","Salary"]] = scaler.transform(data[["Age","Salary"]]);
# print(data);

# Min-max scale the 'Age' and 'Salary' columns

min_max_scaler = preprocessing.MinMaxScaler()

data[["Age", "Salary"]] = min_max_scaler.fit_transform(data[["Age", "Salary"]])

print("\n\n",data);