
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import math;
import statsmodels.api as sm

# Importing the Dataset

dataset = pd.read_csv("CarPrice_Prediction.csv");

# Data Cleaning 

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # Remove leading/trailing spaces in all string columns

# print(dataset.info());
dataset.drop(dataset.columns[[0, 2]], axis=1, inplace=True);

numerical_cols = [
    "symboling", 
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg",
]

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
       
        data.loc[data[c] < lowerBound, c] = int(lowerBound)
        data.loc[data[c] > upperBound, c] = int(upperBound)
    
    return data

dataset = removing_outliers_IQR(dataset,numerical_cols);




# # Visualization Of Dataset

# def datasetVisualization(cols,y_axis,data):
#         for c in cols:
#             plt.scatter(data[c],data[y_axis]);
#             plt.title("Data Visualization");
#             plt.xlabel(c);
#             plt.ylabel(y_axis);
#             plt.grid(True);
#             plt.show();

# # datasetVisualization(numerical_cols,"price",dataset);

# Building a Modal (Bidirectional Elimination)


def bidirectional_selection(X, y, significance_level=0.05):
    initial_features = []
    best_features = list(initial_features)

    # Convert all to numeric and drop NaNs
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    mask = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[mask]
    y = y.loc[mask]

    while True:
        changed = False

        # ---------- Forward Step ----------
        excluded = list(set(X.columns) - set(best_features))
        new_pval = pd.Series(index=excluded, dtype=float)

        for new_column in excluded:
            X_model = sm.add_constant(X[best_features + [new_column]].astype(float))
            model = sm.OLS(y, X_model).fit()
            new_pval[new_column] = model.pvalues.get(new_column, 1.0)

        if not new_pval.empty:
            min_pval = new_pval.min()
            if min_pval < significance_level:
                best_features.append(new_pval.idxmin())
                changed = True

        # ---------- Backward Step ----------
        if best_features:
            X_model = sm.add_constant(X[best_features].astype(float))
            model = sm.OLS(y, X_model).fit()
            pvalues = model.pvalues.iloc[1:]  # exclude intercept
            max_pval = pvalues.max()
            if max_pval > significance_level:
                worst_feature = pvalues.idxmax()
                best_features.remove(worst_feature)
                changed = True

        if not changed:
            break

    return best_features

x_p = dataset.drop("price", axis=1)
x_p = pd.get_dummies(x_p, drop_first=True, dtype=int)
y_p = dataset["price"]

selected_features = bidirectional_selection(x_p, y_p, 0.05)

x_p = x_p[selected_features];


# Ways to Check Multicollinearity Before Modal Training

# print("\n",x_p.corr(numeric_only=True));

x_p = x_p.drop(["cylindernumber_two","stroke","peakrpm","curbweight","cylindernumber_four"], axis=1);

from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---- VIF ----

vif_data = pd.DataFrame()
vif_data["Feature"] = x_p.columns
vif_data["VIF"] = [variance_inflation_factor(x_p.values, i) for i in range(x_p.shape[1])]
print("\nVariance Inflation Factor (VIF):\n", vif_data)

# ---- Tolerance ----
tolerance_data = []
for feature, vif in zip(x_p.columns, vif_data["VIF"]):
    tolerance_data.append((feature, 1/vif))

print("\nTolerance values:")
for feature, tol in tolerance_data:
    print(f"{feature}: {tol}")

# ---- Condition Number ----
correlationMatrix = x_p.corr().values
cond_number = np.linalg.cond(correlationMatrix)
print("\nCondition Number:", cond_number)

# Splitting DataSet into Independent Variables (x) and Dependent Variables (y)

x = x_p.values;
y = dataset.iloc[:,-1].values;

# Splitting the Dataset into the Training Set and Test Set

from sklearn.model_selection import train_test_split;

x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=0);


# Training the Multiple Linear Regression Model on the Training Set

from sklearn.linear_model import LinearRegression;

regressor = LinearRegression();
regressor.fit(x_train,y_train);

# Predicting the Test Set Results

np.set_printoptions(precision=2);

y_predict = regressor.predict(x_test);

test_output = np.concatenate((y_predict.reshape(len(y_predict),1), y_test.reshape(len(y_test),1)), axis=1);

print("\n\n",test_output);

# Parameters / Methods to Check Suitability of MLR

x_predict = regressor.predict(x);

x_train_predict = regressor.predict(x_train);

x_test_predict = regressor.predict(x_test);

from sklearn.metrics import r2_score;

r2 = r2_score(y,x_predict);

r2_train = r2_score(y_train,x_train_predict);

r2_test = r2_score(y_test,x_test_predict);

print("\nR² (Coefficient of Determination) X and Y : ",r2); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Train and Y_Train : ",r2_train); # R² (Coefficient of Determination)

print("\nR² (Coefficient of Determination) X_Test and Y_Test : ",r2_test); # R² (Coefficient of Determination)


print("\nAdjusted R² w.r.t X and Y : ",1 - (1 - r2) * (len(y)  - 1) / (len(y)  - x.shape[1] - 1)); # Adjusted  R² 

print("\nAdjusted R² w.r.t X_Train and Y_Train : ",1 - (1 - r2_train) * (len(y_train)  - 1) / (len(y_train)  - x_train.shape[1] - 1)); # Adjusted  R² 

print("\nAdjusted R² w.r.t X_Test and Y_Test : ",1 - (1 - r2_test) * (len(y_test)  - 1) / (len(y_test)  - x_test.shape[1] - 1)); # Adjusted  R² 


from sklearn.metrics import mean_squared_error;

mse_x = mean_squared_error(y,x_predict); # Mean Squared Error (MSE)

rmse_x = np.sqrt(mse_x);

mse_x_test = mean_squared_error(y_test,x_test_predict);

rmse_x_test = np.sqrt(mse_x_test);

mse_x_train = mean_squared_error(y_train,x_train_predict);

rmse_x_train = np.sqrt(mse_x_train);


print("\nRoot Mean Squared Error (MSE) X and Y : ",rmse_x); # Root Mean Squared Error (MSE)

print("\nRoot Mean Squared Error (MSE)  X_Train and Y_Train : ",rmse_x_train); # Root Mean Squared Error (MSE)

print("\nRoot Mean Squared Error (MSE)  X_Test and Y_Test : ",rmse_x_test); # Root Mean Squared Error (MSE)


print("\nComparing RMSE to the Mean of Y in % : ",((rmse_x*100)/np.mean(y))); # Comparing RMSE to the Mean

print("\nComparing RMSE to the Mean of Y_Train in % : ",((rmse_x_train*100)/np.mean(y_train))); # Comparing RMSE to the Mean

print("\nComparing RMSE to the Mean of Y_Test in % : ",((rmse_x_test*100)/np.mean(y_test))); # Comparing RMSE to the Mean


print("\nMean Absolute Error X and Y: ",np.mean(np.mean(np.abs(x_predict-y))));

print("\nMean Absolute Error X_Train and Y_Train : ",np.mean(np.mean(np.abs(x_train_predict-y_train))));

print("\nMean Absolute Error X_Test and Y_Test: ",np.mean(np.mean(np.abs(x_test_predict-y_test))));


# Homoscedasticity Check

residuals = y - regressor.predict(x);

plt.scatter(regressor.predict(x), residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Homoscedasticity Check")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.grid(True);
plt.show()

print("\nResult : This Dataset and Modal is Moderate for Multiple Linear Regression");