
# Importing the Libraries
import pandas as pd;
import numpy as np;

# Importing the Dataset

dataset = pd.read_csv("Data.csv");

x = dataset.iloc[:, :-1].values;
y = dataset["Purchased"].values;

print(x)
print("\n",y);

# Taking care of missing Data

from sklearn.impute import SimpleImputer;

missingValues = SimpleImputer(missing_values=np.nan, strategy='mean');
x[:,1:]=missingValues.fit_transform(x[:,1:]);

# Encoding the Independent Variable

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("Country Encoding", OneHotEncoder(), [0])
    ],
    remainder="passthrough"
);

x = np.array(ct.fit_transform(x));

print("\n\n",x);

# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder;

le = LabelEncoder();
y=le.fit_transform(y);

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);

print("\n\n",x_train);
print("\n\n",x_test)
print("\n\n",y_train);
print("\n\n",y_test);

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:,3:])

x_test[:,3:] = sc.transform(x_test[:,3:])

print("\n",x_train,"\n\n",x_test)