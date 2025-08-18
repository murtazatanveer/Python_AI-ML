import pandas as pd
from scipy import stats as scipy


# Sample dataset
data = {
    "Name": ["Ali", "Sara", "Ahmed", "Zara", "Omar", "Hina", "Usman", "Ayesha", "Bilal", "Fatima"],
    "Age": [23, 25, 22, 28, 24, 26, 27, 23, 29, 30],
    "Salary": [50000, 60000, 55000, 65000, 58000, 62000, 70000, 52000, 68000, 72000],
    "Marks": [88, 92, 85, 95, 90, 91, 87, 89, 93, 96]
}

stu = pd.DataFrame(data)

# Printing Students Records
print("\nStudents Records:\n", stu);

# Mean , Median, and Mode of Salary
mean_salary = stu["Salary"].mean()
median_salary = stu["Salary"].median()
mode_salary = stu["Salary"].mode()[0]  # mode() returns a Series

print("\nMean Salary:", mean_salary)
print("Median Salary:", median_salary)
print("Mode Salary:", mode_salary)

# Standard Deviation and Variance of Marks

print("\nSample Variance of Marks:", stu["Marks"].var());
print("\nSample Standard Deviation of Marks:", stu["Marks"].std());

print("\nPopulation Variance of Marks:", stu["Marks"].var(ddof=0));
print("\nPopulation Standard Deviation of Marks:", stu["Marks"].std(ddof=0));

# Percentile , Quartiles and Identyfying Outliers

# stu=stu.sort_values(by="Marks");

# print(stu);

Q1 = stu["Marks"].quantile(0.25)
Q3 = stu["Marks"].quantile(0.75)

IQR = Q3 - Q1;

lower_bound = Q1 - (1.5 * IQR)
upper_bond = Q3 + (1.5 * IQR)

print("Q1 (25th percentile):", Q1);
print("Q3 (75th percentile):", Q3);
print("IQR (Interquartile Range):", IQR);
print("Lower Bound for Outliers:", lower_bound);
print("Upper Bound for Outliers:", upper_bond);

print("Outliers in Marks:\n", stu[(stu["Marks"] < lower_bound) | (stu["Marks"] > upper_bond)]);

# Calculating Z-scores for Marks
stu["Marks_Z-Score"] = scipy.zscore(stu["Marks"],ddof=1);
print("\nZ-scores for Marks:\n", stu[["Name", "Marks", "Marks_Z-Score"]]);

# Mean Absolute Deviation (MAD) for Marks

data_values = [160, 165, 170, 175, 180]
series = pd.Series(data_values)

mad = series.sub(series.mean()).abs().mean()   # (|x - mean|).mean()
print("Mean Absolute Deviation (MAD):", mad)

# Corelation

correlation = stu.corr(numeric_only=True);
print("\nCorrelation Matrix:\n", correlation);

d = {"x":[10,15,20,25,30],
     "y":[100,90,80,70,75]
     }

df = pd.DataFrame(d);

print(df.corr(numeric_only=True));