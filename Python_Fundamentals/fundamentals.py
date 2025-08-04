# Factorial of a Number

def factorial(num):
    fact=1;

    for i in range(1,num+1):
        fact = fact*i

    return fact;

print(factorial(5));
print(factorial(10));

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

ran = range(2,10,2);

print(ran[1],type(ran));

oddNumbers = [1,3,5,7,9,11];

for index,num in enumerate(oddNumbers):
    print(index," : ",num)

for number in [2,4,6,7]:
    print(number);

def outer():
    def inner():
        print("I'm inner!")

    inner()  # ✅ Calling inner from inside outer

outer()  # ✅ Output: I'm inner!

# Object Oriented Programming

class Student:

    universityName = "COMSTAS"
    location = "Abbottabad"

    def __init__(self,name,age,marks):
        self.name = name;
        self.age = age;
        self.marks = marks;
        

    def display(self):
            print("\n\nName : ",self.name);
            print("Age : ",self.age);
            print("Marks : ",self.marks);
            print("University : ",self.universityName);
            print("Location : ",self.location,"\n\n");
            
   

s1 = Student("Murtaza", 22, 85);

s1.location="Islamabad"

s2 = Student("Ali", 22, 90);

s1.display();
s2.display();

Student.location = "Peshawar"

s1.display()
s2.display()

s1.location="Abbottabad"

Student.location="Lahore"

s1.display();
s2.display();

# del(s1)

# s1.display();

class Car:

    @classmethod
    def initilizer(cls,name, model):
        cls.name = name;
        cls.model = model
        
    
    def display(self):
            print("\n",self.name,"\t",self.model,"\n");

    @staticmethod
    def start():
            print("Car Started");

c1 = Car();

c1.initilizer("Honda Civic",2019);

c2 = Car();

c1.display();
c2.display();
c2.name = "Suzuki Swift";

c2.initilizer("Honda City",2010);

c1.display();
c2.display();

# def d(x):print(x);

# d(); # error

class A:
     
    def __init__(self,x):
        print("A's constructor called");
        self.__x = x;
        print("A's constructor completed");

    def display(self):
        print("Value of x in Class A is ",self.x);

class B(A):
     
    def __init__(self,y):
        print("B's constructor called");
        self.__y = y;
        super().__init__(y*2)  # Call A's constructor
        print("B's constructor completed");

    def display(self):
        print("Value of y in Class B is ",self.y);
        super().display();

ob = B(10);
# ob.display();
# print(ob.__dict__);  # Error X and Y are private attributes


class Calculator:

    def __init__(self, a, b):
        print("Calculator Initialized");
        self.a = a
        self.b = b
    
    @property
    def add(self):
        return self.a + self.b
    
    @property
    def sub(self):
        return self.a-self.b
    
    @property
    def mul(self):
        return self.a*self.b
    
    @property
    def div(self):
        return self.a/self.b
    
calc = Calculator(10, 5)

print("Addition:", calc.add)
print("Subtraction:", calc.sub)
print("Multiplication:", calc.mul)
print("Division:", calc.div)


class ComplexNumber:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real, self.imag + other.imag);
    
n1 = ComplexNumber(5,8);
n2 = ComplexNumber(2,3);

add = n1+n2;

print(add.__dict__)


# class A:
    
#     def __eq__(self,other):
#         print("Class A eq Called");
#         return True;

# class B:
    
#     def __eq__(self,other):
#         print("Class B eq Called");
#         return True;
# a=A();
# b=B();

# print(a==b);


# import matplotlib.pyplot as plt

# x = [1, 2, 3, 4]
# y = [2, 4, 6, 8]

# plt.plot(x, y)
# plt.title("Simple Line Graph")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show()
