import numpy as np;
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
