import pandas as pd;
import numpy as np;
from sklearn import preprocessing;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;

# Load the dataset

data = pd.read_csv("Data.csv");

# Fill the Missing Values using Simple Imputer

# fillMissingValues = SimpleImputer(strategy="mean")
# data[["Age","Salary"]] = fillMissingValues.fit_transform(data[["Age","Salary"]]);

# print(data)

# Standardize the 'Age' and 'Salary' columns

# scaler = preprocessing.StandardScaler();

# scaler.fit(data[["Age","Salary"]]);
# data[["Age","Salary"]] = scaler.transform(data[["Age","Salary"]]);
# print(data);

# Min-max scale the 'Age' and 'Salary' columns

# min_max_scaler = preprocessing.MinMaxScaler()

# data[["Age", "Salary"]] = min_max_scaler.fit_transform(data[["Age", "Salary"]])

# print("\n\n",data);

# OneHotEncoder in Sklearn

# encoder = preprocessing.OneHotEncoder();
# countryEncoding = encoder.fit_transform(data[["Country"]]);

# Concept of Transformers in Sklearn

# ct = ColumnTransformer(

#     transformers=[
#         ("Missing Values",SimpleImputer(strategy="mean"),["Age","Salary"]),
#         ("Country Encoding",preprocessing.OneHotEncoder(),["Country"])
#     ],
#     remainder="passthrough"
# )

# data = np.array(ct.fit_transform(data));

# print(data,"\n\n");

# RobustScalar in Sklearn

stu = {
    "ID": range(1, 11),
    "Name": ["Ali", "Sara", "Umar", "Hina", "Ahmed", "Ayesha", "Bilal", "Zara", "Hassan", "Fatima"],
    "Age": [22, 27, 24, 30, 28, 23, 26, 35, 21, 29],
    "City": ["Karachi", "Lahore", "Karachi", "Islamabad", "Karachi", "Lahore", "Islamabad", "Karachi", "Lahore", "Karachi"],
    "Salary": [50000, 60000, 55000, 70000, 65000, 52000, 72000, 80000, 58000, 500000],
    "Department": ["HR", "IT", None, "Finance", "IT", "Finance", "HR", "Finance", None, "IT"],
    "Join_Date": ["2022-05-10", "2023-02-15", "2021-12-01", "2024-06-20", "2023-05-05", "2022-08-08", "2024-01-10", "2021-09-15", "2023-12-25", "2022-11-30"]
}

stu = pd.DataFrame(stu)
stu.set_index("ID", inplace=True)

robust_scaler = preprocessing.RobustScaler();
outliars = robust_scaler.fit_transform(stu[["Salary"]]);
print(outliars);
