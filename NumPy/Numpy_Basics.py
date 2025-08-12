import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import time;

# Creating Array From List

arr_1d = np.array([0,1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15]);
print(arr_1d);

arr_2d = np.array([[1, 2, 3], [4, 5, 6],["abc",False, 7.5]]);
print(arr_2d);

# List VS Numpy Array Operation

list_1d = [1, 2, 3, 4, 5];

print("Python List Multiplication : ", list_1d*2);
print("Numpy Array Multiplication : ", arr_1d*2);

start = time.time();

lst = list(i*2 for i in range(1000000));

print("List Operation Time : ", time.time()-start);

start = time.time();

arr = np.arange(1000000)*2;

print("Numpy Array Operation Time : ", time.time()-start);

# Creating Array From Scratch Using Numpy Builtin Functions

zeros = np.zeros((3,4));

print("Zeros Array : ", zeros);

ones = np.ones((2,3));

print("Ones Array : ", ones);

full = np.full((5,2), 99);
print("Full Array : ", full);

random = np.random.randint(1,101, size=(7, 4));
print("Random Array : ", random);

tensor = np.array([[[1,2],[3,4]],
                    [[5,6],[7,8]],
                    [[9,10],[11,12]]]);

print("Tensor Array : ", tensor);

# Array Properties

print("Shape of 1D Array : ", arr_1d.shape);
print("Shape of 2D Array : ", arr_2d.shape);
print("Shape of Tensor Array : ", tensor.shape);

print("Dimension of 1D Array : ", arr_1d.ndim);
print("Dimension of 2D Array : ", arr_2d.ndim);
print("Dimension of Tensor Array : ", tensor.ndim);

print("Size of 1D Array : ", arr_1d.size);
print("Size of 2D Array : ", arr_2d.size);
print("Size of Tensor Array : ", tensor.size);

print("Data Type of 1D Array : ", arr_1d.dtype);
print("Data Type of 2D Array : ", arr_2d.dtype);
print("Data Type of Tensor Array : ", tensor.dtype);

# Array Reshaping 

arr = np.arange(1, 13);
print("Original Array : ", arr);

reshaped = arr.reshape(3,4);
print("Reshaped Array : ", reshaped);

flatten = arr.flatten();
print("Flattened Array : ", flatten);

ravelled = flatten.ravel(); # Ravel returns a flattened array, but it may return a view of the original array if possible.
print("Ravelled Array : ", ravelled);

transposed = reshaped.T;
print("Transposed Array : ", transposed);


# Numpy Array Operations

print("Addition of 1D Array with Scalar : ", arr_1d + 10);
print("Subtraction of 1D Array with Scalar : ", arr_1d - 2);
print("Multiplication of 1D Array with Scalar : ", arr_1d * 3);
print("Division of 1D Array with Scalar : ", arr_1d / 2);

print("Basic Slicing : ",arr_1d[3:7]);
print("Slicing With Step : ",arr_1d[3:13:3]);

print(arr_2d,type(arr_2d));

print("Specific Element : ",arr_2d[2,0]);
print("Row Slicing : ", arr_2d[1,:]);    
print("Column Slicing : ", arr_2d[:,1]); 

# Sorting

unsorted = np.array([-2, 3, 15, 1, 11, 4, -5, 17, 2, 13, 5, -3, 9, -7, 22, 0]);

sortedArr = np.sort(unsorted);
print("Unsorted Array : ", unsorted);    
print("Sorted Array : \n", sortedArr);

unsorted_2d = unsorted.reshape(4, 4);

print(unsorted_2d);

sorted_2d = np.sort(unsorted_2d, axis=0);  
print("Sorted 2D Array : \n", sorted_2d);

# Filter

numbers = np.arange(1, 31);
even_numbers = numbers[numbers%2==0];

print("Even Numbers : ", even_numbers);

# Filter With Mask

maskFiveMultiples = numbers%5==0
fiveMultiples = numbers[maskFiveMultiples];
print("Multiples of 5 : ", fiveMultiples);

# Fancy Indexing VS np.where()

print("\n\nOriginal Numbers Array : ", numbers);

indices =[3,6,1,2];
fancyIndexed = numbers[indices];
print("Fancy Indexed Array : ", fancyIndexed);
print("Np Where : ",np.where(numbers%5==0));

product = np.where(numbers%5==0, numbers*numbers, numbers);
print(product);

# Array Concatenation

oneTOTwenty = np.concatenate((np.arange(1,11),np.arange(11,21)));

print(oneTOTwenty);

# Array Compatibility

a = np.array([1,2,3]);
b = np.array([4,5,6]);
c = np.array([7,8,9]);

print("Array A Shape : ", a.shape,"Array A Dimension : ", a.ndim);
print("Array B Shape : ", b.shape,"Array B Dimension : ", b.ndim);
print("Array C Shape : ", c.shape,"Array C Dimension : ", c.ndim);

print("Array Compatibility Check : ", a.shape == b.shape == c.shape);
print("Array Dimesions Check : ", a.ndim == b.ndim == c.ndim);

# Array Stacking

print("\n\n",reshaped,"\n");

a1 = np.array([[1,2],[3,4]]);
a2 = np.array([[5,6],[7,8]]);

stack = np.stack((a1,a2));

print("\nStacked Array : \n",stack); # In Stack shape and dtype of both 2D Arrays must be same.

verStack = np.vstack((a1,a2));
print("\nVertically Stacked Array : \n", verStack);

horStack = np.hstack((a1,a2));
print("\nHorizontally Stacked Array : \n", horStack);


# Adding Element to 1D Array

arr_1d = np.append(arr_1d,[66,77,88,99]);
print("\n\nUpdated 1D Array : ", arr_1d);

# print(pd.DataFrame(reshaped , columns=["A","B","C","D"] , index=["I1","I2","I3"])); # Converting Numpy Array to Dataframe

# Deletion in Numpy Array

print("\n1D Array Before Deletion : ", arr_1d);

arr_1d = np.delete(arr_1d,[0,2,4]);

print("\n1D Array After Deletion : ", arr_1d);

print("\n2D Array Before Deletion : \n", reshaped);

reshaped = np.delete(reshaped,[0,2],axis=1);

print("\n2D Array After Deletion : \n", reshaped);


# Practicing Numpy Arrays

arr_2d = np.array([
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

print("\nArray 2D : \n", arr_2d);

print(arr_2d[:2,:3]);

print("\nMiddle Elements : \n",arr_2d[2:4, 1:4]);


# Analyzing Sales Data

sales_data = np.array([
    [1, 150000, 180000, 220000, 250000],
    [2, 120000, 140000, 160000, 190000],
    [3, 200000, 230000, 260000, 300000],
    [4, 180000, 210000, 240000, 270000],
    [5, 160000, 185000, 205000, 230000]
])

print("\n\nSample Data For First 3 Resturants : ",sales_data[:3,1:]);

print("Total Sales For Each Resturant : ",sales_data[:,1:].sum(axis=1));

average_sales = sales_data[:,1:].mean(axis=1);

print("Average Sales For Each Resturant : ",average_sales);

print("Maximum Sales For Each Resturant : ",sales_data[:,1:].max(axis=1));

print("Minimum Sales For Each Resturant : ",sales_data[:,1:].min(axis=1));


print("Cumsum Sales For Each Resturant : ",sales_data[:,1:].cumsum(axis=1));

print("Sales Data Shape : ", sales_data.shape)

# plt.figure(figsize=(10, 6));
# plt.plot(average_sales);
# plt.title('Average Sales Per Resturant');
# plt.xlabel('Resturant')
# plt.ylabel('Average Sales');
# plt.grid(True);
# plt.show();

# Matrix Operation 

m1 = np.array([[1, 2, 3],
                [4, 5, 6 ]]);

m2 = np.array([[7, 8, 9],
                [10, 11, 12]]);

print("Matrix Multiplication : \n", m1*m2);

print("Matrix Addition : \n", m1+m2);

print("Matrix Subtraction : \n", m1-m2);

print("Matrix Division : \n", m1/m2); 

print("Matrix Dot Product : \n", np.dot(m1, m2.T));  # Transpose of m2 is used for dot product

# Converting Strings array to Upper Case

months = np.array(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]);

print("Months in Upper Case : ", np.char.upper(months));

# Average Sales Per Month

sales_per_month = sales_data[:, 1:]/12;
print("\nAverage Sales Per Month : ", sales_per_month);

# Saving and Loading Arrays

np.save("sales_data.npy", sales_data);

loaded_sales_data = np.load("sales_data.npy");
print("\nLoaded Sales Data : \n", loaded_sales_data);

# Setting Custom Dimensions

arr = np.array([[[1, 2, 3, 4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]], ndmin=10)

print(arr)
print('\n\nNumber Of Dimensions :', arr.ndim)

# Difference Between Copy and View

a = np.array([1, 2, 3, 4])
view = a[1:3]   # This is a view
view[0] = 99    # Modifies original
print("View :", view)  # [99  3]
print("a:", a)  # [ 1 99  3  4]

a= np.array([1, 2, 3, 4])
copy = a[1:3].copy()  # Explicit copy
copy[0] = 99          # Does not affect original
print("Copy :", copy)  # [99  3]
print("a:", a)  # [1  2  3  4]

print(np.array([[1,2,3],[4,5,6]]).shape)
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("\n");

# Numpy Array Iteration 

for x in arr_1d:
    print(x, end=" ");
print("\n");

for row in arr_2d:
        print(row);

print("\n");

for element in arr_2d.flat:
      
    print(element, end=" ");

print("\n");


for element in np.nditer(arr_2d , order='F'):  # 'F' for Fortran-style (column-major) order
    print(element, end=" ");

for index,element in np.ndenumerate(arr_1d):
    print(f"Index: {index}, Element: {element}",);


# Advance Indexing Methods

print("\n\nAdvanced Indexing Methods\n");
abc = np.array([[1, 2, 3],
                 [4, 5, 6]])

result = abc[(0, 1,0), (2, 1,1)]
print(result) # Output: [3 5 2]

# Splitting NumPy Arrays

nums_1d = np.array([2,4,6,8,10,12,14,16,18,20]);


nums_2d = np.array([ [1,3,5,11],
                     [7,9,11,22],
                    [13,15,17,33],
                    [19,21,23,44],
                    [25,27,29,55],
                    [31,33,35,66],]);

split_1d = np.array_split(nums_1d, 4);
print("\n1D Array Split : ", split_1d);

split_2d = np.array_split(nums_2d,2);
print("\n2D Array Split : ", split_2d);

split_2d_col = np.array_split(nums_2d, 3, axis=1);
print("\n2D Array Split By Column : ", split_2d_col);

# Searching in Numpy Arrays

print("\nSearching in Numpy Arrays\n");
print("\nSeaching in 1D Array : ", np.where(nums_1d==14)[0][0]);

isElevenPresent = np.where(nums_2d==11);
print("\nSearching in 2D Array : ","(", isElevenPresent[0][0],",",isElevenPresent[1][0],")");

# Binary Search in Numpy Arrays
print("\nBinary Search in Numpy Arrays\n")

print("Binary Search in 1D Array : ", np.searchsorted(nums_1d, 14));

y = np.array([[0,1,2],[3,4,5],[6,7,8]]).flatten();

print("Binary Search in 2D Array : ", np.searchsorted(y, 9));