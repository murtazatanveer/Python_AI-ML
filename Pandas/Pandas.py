import pandas as pd
import matplotlib.pyplot as plt
import numpy as np;

# DataFrame in Pandas

# school = {
#     "students":["Alice", "Bob", "Charlie"],
#     "subjects":["Math", "Science", "History"],
#     "grades":[85, 90, 88],
#     "ages":[15, 16, 19]
# }

# output = pd.DataFrame(school,index=["Stu 1","Stu 2","Stu 3"]);

# print(output);
# print(output.loc[["Stu 1","Stu 3"]]);

# print(type (output) , output.describe());

# print(output["students"]);
# print(output.iloc[1]);


# Series in Pandas

# a = [1, 7, 2]

# ser = pd.Series(a)

# print(ser)
# print(type(ser))

# name = ["Alice", "Bob", "Charlie"]
# marks = [85, 90, 88]

# res = pd.Series(marks, index=name);
# print(res,res["Alice"]);

# # print(type(res.values));

# stu = {
#     "name":["John", "Charlie", "Doe"],
#     "marks":[85, 90, 88],
#     }

# print(pd.Series(stu).values);

# DataFrame and CSV file

# s = pd.read_csv("students.csv");

# print(s.loc[list(range(3,7))+[9,1,11]]);

# print(type(s.loc[0]));


# js = pd.read_json("students.json");
# print(js);

# print(js.head(5));

# print(js.loc[list(range(2,6))+[0,9,11]]);

# print(js.tail(7));

# print(js.info());


# Cleaning Data Using Pandas

# pulse = pd.read_csv("pulse.csv");

# print(pulse.iloc[0:10]);

# print(pulse.iloc[0:5, 0]);

# cleaned_pulse = pulse.dropna();

# print(pulse["Duration"])

# print(pulse);

# print(cleaned_pulse);

# print(pulse.dropna(axis=1));

# pulse.fillna("HELLO", inplace=True);
# print(pulse);

# pulse.fillna(pulse["Calories"].mean(), inplace=True);
# pulse["Calories"]=pulse["Calories"].astype(int);
# print(pulse.info());
# print(pulse);

# pulse["Avearge Pulse"]=pulse["Pulse"]+ pulse["Maxpulse"] / 2;

# print(pulse);

# print(pulse.describe());

# pulse.fillna(pulse["Calories"].median(), inplace=True);

# pulse.fillna(pulse["Calories"].mode()[0], inplace=True);

# pulse["Date"]=pd.to_datetime(pulse["Date"], format="mixed");

# print(pulse);

# print(pulse.loc[2,"Calories"]);

# pulse.drop(["Duration","Maxpulse"],inplace=True,axis=1);

# pulse.drop(pulse.index[[0,1,2]],inplace=True);
# print(pulse.iloc[0:5,0:2]);

# print(pulse.duplicated());

# pulse.drop_duplicates(inplace=True);

# print(pulse);

# Corelation in Pandas

# scores = {
#     "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
#     "Math": [85, 90, 88, 92, 95],
#     "Science": [78, 82, 85, 89, 91],
#     "History": [80, 85, 84, 87, 90]
# }

# scores_df = pd.DataFrame(scores)

# print(scores_df.corr(numeric_only=True));

# Pandas Plotting

# stu = {
#     "names":["Alice", "Bob", "Charlie", "David", "Eve"],
#     "cgpa":[3.5, 3.8, 3.6, 3.2, 2.2],
# }

# stu = pd.DataFrame(stu);

# stu.plot(kind="line", x="names", y="cgpa", title="CGPA of Students", xlabel="Names", ylabel="CGPA");
# plt.show();

# print(stu.describe());

# pulse.to_csv("export.csv",index=False);

# DataFrame 1: Student basic info

# df1 = pd.DataFrame({
#     'ID': [101, 102, 103],
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [20, 21, 22]
# })

# # DataFrame 2: Student academic info
# df2 = pd.DataFrame({
#     'ID': [101, 102, 104],
#     'CGPA': [3.5, 3.7, 3.9],
#     'Grade': [["A+","A","C+"], ["F","A","B+"], ["A","C+","B"]]
# })

# df1.info();
# concat = pd.concat([df1, df2],axis=0);
# merged_df = pd.merge(df1, df2, on='ID', how='inner');
# print(concat,"\n\n");
# print(merged_df);



# --------------------------
# 1. CREATE SAMPLE DATAFRAME
# --------------------------
# data = {
#     "ID": range(1, 11),
#     "Name": ["Ali", "Sara", "Umar", "Hina", "Ahmed", "Ayesha", "Bilal", "Zara", "Hassan", "Fatima"],
#     "Age": [22, 27, 24, 30, 28, 23, 26, 35, 21, 29],
#     "City": ["Karachi", "Lahore", "Karachi", "Islamabad", "Karachi", "Lahore", "Islamabad", "Karachi", "Lahore", "Karachi"],
#     "Salary": [50000, 60000, 55000, 70000, 65000, 52000, 72000, 80000, 58000, 75000],
#     "Department": ["HR", "IT", None, "Finance", "IT", "Finance", "HR", "Finance", None, "IT"],
#     "Join_Date": ["2022-05-10", "2023-02-15", "2021-12-01", "2024-06-20", "2023-05-05", "2022-08-08", "2024-01-10", "2021-09-15", "2023-12-25", "2022-11-30"]
# }

# df = pd.DataFrame(data)
# print("\nInitial DataFrame:\n", df)

# --------------------------
# 2. INDEX MANIPULATION
# --------------------------
# df.set_index("ID", inplace=True)  # Change index to "ID"
# print("\nIndex changed to 'ID':\n", df)

# df.reset_index(inplace=True)  # Reset to default numeric index
# print("\nIndex reset to default:\n", df)

# # --------------------------
# # 3. CONDITIONAL SELECTION
# # --------------------------
# cond1 = df[(df["Age"] > 25) & (df["City"] == "Karachi")]
# print("\nRows where Age > 25 and City == Karachi:\n", cond1)

# cond2 = df[df["Salary"] > df["Salary"].mean()]
# print("\nRows where Salary > mean salary:\n", cond2)

# # --------------------------
# # 4. ADDING & MODIFYING COLUMNS
# # --------------------------
# df["Bonus"] = df["Salary"] * 0.10
# print("\nAdded 'Bonus' column:\n", df)

# df["Department"].fillna("Unknown", inplace=True)
# print("\nMissing Department values filled:\n", df)

# # --------------------------
# # 5. DROPPING DATA
# # --------------------------
# df.drop(columns="Bonus", inplace=True)
# print("\nDropped 'Bonus' column:\n", df)

# df.drop(index=[1, 3, 5], inplace=True)
# print("\nDropped rows with index 1, 3, 5:\n", df)

# # --------------------------
# # 6. GROUPBY & AGGREGATION
# # --------------------------
# avg_salary = df.groupby("Department")["Salary"].mean()
# print("\nAverage Salary by Department:\n", avg_salary)

# city_count = df.groupby("City")["Name"].count()
# print("\nNumber of employees per City:\n", city_count)

# # --------------------------
# # 7. MERGING & JOINING
# # --------------------------
# df2 = pd.DataFrame({
#     "ID": [1, 2, 4, 5, 7, 9],
#     "Experience": [2, 4, 5, 3, 6, 2]
# })

# merged_df = pd.merge(df, df2, on="ID", how="inner")
# print("\nMerged on 'ID':\n", merged_df)

# dept_df = pd.DataFrame({
#     "Department": ["HR", "IT", "Finance", "Unknown"],
#     "Manager": ["Mr. A", "Ms. B", "Mr. C", "Ms. D"]
# })

# left_join_df = pd.merge(df, dept_df, on="Department", how="left")
# print("\nLeft join on Department:\n", left_join_df)

# # --------------------------
# # 8. DATE & TIME
# # --------------------------
# df["Join_Date"] = pd.to_datetime(df["Join_Date"])
# print("\nJoin_Date converted to datetime:\n", df)

# after_date = df[df["Join_Date"] > "2023-01-01"]
# print("\nRows where Join_Date > 2023-01-01:\n", after_date)

# # --------------------------
# # 9. ADVANCED INDEXING
# # --------------------------
# cities_list = ["Karachi", "Lahore"]
# city_filter = df[df["City"].isin(cities_list)]
# print("\nCity is in Karachi or Lahore:\n", city_filter)

# age_between = df[df["Age"].between(25, 35)]
# print("\nAge between 25 and 35:\n", age_between)

# # --------------------------
# # 10. APPLY FUNCTIONS
# # --------------------------
# df["Salary"] = df["Salary"].apply(lambda x: x * 1.05)  # Increase by 5%
# print("\nSalary increased by 5%:\n", df)

# df_upper = df.applymap(lambda x: x.upper() if isinstance(x, str) else x)
# print("\nAll strings to uppercase:\n", df_upper)

# # --------------------------
# # 11. EXPORTING DATA
# # --------------------------
# df.to_csv("students_data.csv", index=False)
# df.to_excel("students_data.xlsx", sheet_name="Student Info", index=False)

# print("\nData saved to 'students_data.csv' and 'students_data.xlsx'")

