import matplotlib.pyplot as plt;
import numpy as np;

x = np.array([1,2,3,4,5,6]);
y_actual = np.array([50,60,80,90,95,98]);
# y = np.array([88,76,81,85,79,89]);

# # Basic Line Plotting

# # plt.plot(x, y, color="orange");
# # plt.title("Basic Line Plotting");

# Drawing a Line of Best Fit (like Regression)

b1 = np.sum((x-np.mean(x))*(y_actual-np.mean(y_actual)))/np.sum((x-np.mean(x))**2);

b0 = np.mean(y_actual) - b1*(np.mean(x));

y_predict = b0 + b1*(x);

x_test = np.array([0.5,2.5,8,9,10,11]);

y_test = b0 + b1*(x_test);

plt.scatter(x, y_actual, color="red", label="Actual Data Points");
plt.plot(x, y_predict, color="green", label="Predicted Linear Line");

plt.xlabel("X-Axis");
plt.ylabel("Y-Axis");
plt.grid(True);
plt.show();

