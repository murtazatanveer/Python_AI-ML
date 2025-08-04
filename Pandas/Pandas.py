import pandas as pd

# DataFrame in Pandas

# school = {
#     "students":["Alice", "Bob", "Charlie"],
#     "subjects":["Math", "Science", "History"],
#     "grades":[85, 90, 88],
#     "ages":[15, 16, 19]
# }

# output = pd.DataFrame(school);

# print(output);

# print(type (output) , output.describe());

# print(output["students"]);
# print(output.iloc[1]);



# Series in Pandas

a = [1, 7, 2]

ser = pd.Series(a)

print(ser)
print(type(ser))

name = ["Alice", "Bob", "Charlie"]
marks = [85, 90, 88]

res = pd.Series(marks, index=name);
print(res,res["Alice"]);

print(type(res.values));
