import pandas as pd

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


js = pd.read_json("students.json");
print(js);