# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

print(x,"\n",y)

# Feature Scaling

from sklearn.preprocessing import StandardScaler;

sc_x = StandardScaler()
sc_y = StandardScaler();

x = sc_x.fit_transform(x);
y = sc_y.fit_transform(y);

# Training the SVR model on the whole dataset

y = y.flatten();

from sklearn.svm import SVR;

regressor = SVR(kernel="rbf");
regressor.fit(x,y);

# Predicting a new result

print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)));

# Visualising the SVR results

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y.reshape(-1,1)), color="red");
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)))
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y.reshape(-1,1)), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()