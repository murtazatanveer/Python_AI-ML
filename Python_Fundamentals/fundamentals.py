# Factorial of a Number

num = 5;
fact=1;

for i in range(1,num+1):
    fact = fact*i

print("Factorial of ",num," is ",fact);

print(type(fact));


# Datatypes in Python

number = 100; #Integer
decimalNumber = 10.259; #Float
name = "MURTAZA";#String
isGood = False; #Boolean
noneType = None; #None

print(type(noneType));

print(str(type(True and True))+" PYTHON",not True);

# Sum of two Numbers

a,b=2,10.52;

print(a+b);

#Type Casting 

num1 , num2 = "10.52" , 2;

print(float(num1)+num2);


# n = int(input("Enter Any Number : "));

# print(n);

# Slicing in Strings

lan = "I am Learning Python Fundamentals"

print(lan[5:]);
print(lan[:13]);
print(lan[14:20]);

# Ternary Operator 
age = 18
status = "Adult" if age >= 18 else "Minor"
print(status)  # Output: Adult

# Triangle Type Checker

s1 = 25;
s2=10;
s3 = 20;

isValid = True if ((s1+s2)>s3 and (s1+s3)>s2 and (s2+s3)>s1 ) else False;

if(isValid):
    if(s1==s2==s3):
        print("Equilateral Triangle");
    elif(((s1==s2) and (s3!=s1)) or ((s1==s3) and (s2!=s1)) or ((s3==s2) and (s1!=s3))):
        print("Isosceles Triangle");
    else:
        print("Scalene Triangle");
else:
    print("Triangle Is Not Valid");

# Strings Split

words = lan.split();
print(words);

# Lists in Python

evenNumbers = [2,4,6,8,10];

carAndModel=["Toyota",2010,"Civic",2015,"Swift",2024,"Alto",2022,"Wagon R",2023];

print(evenNumbers,carAndModel);

hondaCity = ["Honda City",2021];

carAndModel.extend(hondaCity);

print(carAndModel);

lst = [];

print(type(lst));

tup = (1,5,8,3);

print(tup[1]);

print(type(tup[2:]));


l1 = [1,2,3];
l2 = [1,2,4];

print(l1==l2);


# Disctionaries

student = {
    "name" :"xyz",
    "class": 9,
    "age":16,
    "marks":{
            "eng":55,
            "urdu":60,
            "math":85
    },
}

print(student["marks"]);

student["interest"]=["coding","gym","netflix"];

print(student);

values = student.values();

print(type(values));

print(values);

for v in values:
    print(v);

for key,values in student.items():
    print(key," : ",values);


# # Frequency of Words in a Sentence

# sentence = "python code is fun because python code teaches logic and python code builds logic";

# wordsOfSentence = sentence.split();

# frequency={};

# for w in wordsOfSentence:
    
#     if w in frequency:
#         frequency[w] += 1;
#     else:
#         frequency[w]=1;


# print(frequency , frequency.get("python"));

# # print(frequency.keys());

# # print(frequency.items());

# list1 = [1,2,3,4,5];

# list2 = list1;

# list2.append(99);

# print(list1);  # Output: [1, 2, 3, 4, 5, 99]

# # Sets in Python

# set1 = {1,2,3,4,5,2,1,6,7,5,8,9,10};

# print(set1);  # Output: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# set1.add(());
# print(set1);

# print((1)==());

# setA = {1,2,3,7,11,13,15,18,4,5};
# setB = {1,2,3,4,5};

# print(setA>=setB,setA.isdisjoint(setB));


for i,j,k in [(1,2,3),(11,21,31),(50,60,70)]:
    print(i,j,k);

fruits = ["apple", "banana", "cherry"];

for index, fruit in enumerate(fruits):
    print(index, fruit)