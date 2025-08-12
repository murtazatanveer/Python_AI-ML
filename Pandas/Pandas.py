import pandas as pd
import matplotlib.pyplot as plt

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
# pulse.drop([2,4,6]);
# print(pulse);

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

df1 = pd.DataFrame({
    'ID': [101, 102, 103],
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [20, 21, 22]
})

# DataFrame 2: Student academic info
df2 = pd.DataFrame({
    'ID': [101, 102, 104],
    'CGPA': [3.5, 3.7, 3.9],
    'Grade': [["A+","A","C+"], ["F","A","B+"], ["A","C+","B"]]
})

df1.info();
concat = pd.concat([df1, df2],axis=1);
merged_df = pd.merge(df1, df2, on='ID', how='inner')
print(concat,"\n\n");
print(merged_df);