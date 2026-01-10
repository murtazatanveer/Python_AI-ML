import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

sales = pd.read_csv("Temperature_VS_IceCream.csv");

sales.sort_values(by=["Temperature"], inplace=True, ascending=True);

# Calculating EWMA

ewma = sales["Ice Cream Profits"].ewm(alpha=0.1).mean()

# Plotting the results
plt.scatter(sales["Temperature"], sales["Ice Cream Profits"], label="Original Data", color='red')
plt.plot(sales["Temperature"], ewma, label="EWMA", color='green')
plt.xlabel("Temperature")
plt.ylabel("Ice Cream Profits")
plt.title("EWMA of Ice Cream Profits vs Temperature")
plt.legend()
plt.show()