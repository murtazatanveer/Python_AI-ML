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

ct = ColumnTransformer(

    transformers=[
        ("Missing Values",SimpleImputer(strategy="mean"),["Age","Salary"]),
        ("Country Encoding",preprocessing.OneHotEncoder(),["Country"])
    ],
    remainder="passthrough"
)

data = np.array(ct.fit_transform(data));

print(data,"\n\n");

