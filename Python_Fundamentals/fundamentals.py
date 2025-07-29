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

