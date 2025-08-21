# Importing the Libraries

import pandas as pd;
import numpy as np;
from sklearn import preprocessing;

# Importing the Dataset

dataset = pd.read_csv("centuries.csv", index_col="No.")

# Datatype Conversion and Data Cleaning

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns

dataset["Runs"] = dataset["Runs"].str.replace("*", "", regex=False)

dataset["Runs"] = pd.to_numeric(dataset["Runs"], errors="coerce")

dataset["Result"] = dataset["Result"].str.replace(" (D/L)", "", regex=False);


dataset.loc[79,"Date"] = "05-Nov-23"

dataset["Date"] = pd.to_datetime(dataset["Date"], format="%d-%b-%y");

dataset["Year"] = dataset["Date"].dt.year;
dataset["Month"] = dataset["Date"].dt.month;
dataset["Day"] = dataset["Date"].dt.day;

dataset.drop("Date" , axis=1 , inplace=True);
# dataset.info();

# Total Missing Values In Each Column

# print(dataset.isnull().sum());

# Taking Care of Missing Data

from sklearn.impute import SimpleImputer;

missingValue = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
dataset[["Ground"]] = missingValue.fit_transform(dataset[["Ground"]])

# Handling Duplicates

dataset.drop_duplicates(inplace=True);

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


dataset = removing_outliers_IQR(dataset,["Runs","Position","Innings"]);

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = dataset.drop(dataset.columns[6], axis=1).values; 

y = dataset.iloc[:,-4].values;

print(x[0:5,:]);

# Encoding the Independent Variable 

from sklearn.compose import ColumnTransformer;

ct = ColumnTransformer(
    transformers=[
        ("Encoding the Independent Variable",preprocessing.OneHotEncoder(),[1,4,5])
    ],
    remainder="passthrough"
)

x = ct.fit_transform(x);

# Encoding the Dependent Variable

y[51]="Drawn";
le = preprocessing.LabelEncoder();
y = le.fit_transform(y);

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Feature Scaling 



scaler = preprocessing.StandardScaler(with_mean=False)
x_train[:, -6:] = scaler.fit_transform(x_train[:, -6:]);
x_test[:, -6:] = scaler.transform(x_test[:, -6:]);

