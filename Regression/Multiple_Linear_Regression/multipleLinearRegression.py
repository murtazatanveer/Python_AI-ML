# Importing the Libraries

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

# Importing the Dataset

dataset = pd.read_csv("50_Startups.csv");

x = dataset.iloc[:,:-1].values;
y = dataset.iloc[:,-1].values;


# Encoding the Categorical Data

from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;

ct = ColumnTransformer(

    transformers=[
        ("State Encoding", OneHotEncoder(), [-1])
    ],
    remainder="passthrough"
);

x = np.array(ct.fit_transform(x));


# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);