import numpy as np;

# Linear Algebra Topics

# Scalars, Vectors, Matrices
scalar = 5
vector_1 = np.array([1, 2]);
vector_2 = np.array([4, 5]);

matrix = np.array([ [1, 2],
                    [3, 4],
                    [5, 6]]);

# Matrix Addition and Subtraction

print("\nMatrix Addition and Subtraction\n");

A = np.array([[1,5,9],
              [2,7,4],
              [3,6,8]]);

B = np.array([[2,3,4],
              [5,6,7],
              [8,9,10]]);

Add = A + B;
Sub = A - B;

print("Matrix Addition:\n", Add);
print("Matrix Subtraction:\n", Sub);

# Matrix Multiplication

print("\nMatrix Multiplication\n");

Mul = np.dot(A, B);
print(Mul);

print("\nElement-wise Multiplication:\n", A * B);

# Vectors Dot Product

print("\nVector Dot Product");

dot_product = np.dot(vector_1, vector_2);

print("\nDot Product of vector_1 and vector_2:", dot_product);

# Vectors Cross Product

print("\nVector Cross Product");
cross_product = np.cross(vector_1, vector_2);

print("\nCross Product of vector_1 and vector_2:", cross_product);

# Identity Matrix
print("\nIdentity Matrix");
identity_matrix = np.identity(4);
print("\n4x4 Identity Matrix:\n", identity_matrix);

#Transpose of a Matrix
print("\nTranspose of a Matrix");
transpose_matrix = np.transpose(matrix);
print("\nTranspose of the matrix:\n", transpose_matrix);

# Inverse of 2x2 Matrix

print("\nInverse of a 2x2 Matrix");

matrix_2x2 = np.array([[4, 7],
                       [2, 6]]);

inverse_matrix_2x2 = np.linalg.inv(matrix_2x2);
print("\nInverse of the 2x2 matrix:\n", inverse_matrix_2x2);

# Inverse of 3x3 Matrix

matrix_3x3 = np.array([[1,2,3],
                       [4,5,6],
                       [7,2,9]]);

inverse_matrix_3x3 = np.linalg.inv(matrix_3x3);

print("\nInverse of the 3x3 matrix:\n", inverse_matrix_3x3);


# Vector Norms
print("\nVector Norms");

v = np.array([3, 4, 5,6]);
print("L₂ Norm — Euclidean norm : ",np.linalg.norm(v, ord=2));
print("L₁ Norm — Manhattan norm : ",np.linalg.norm(v, ord=1));
print("L∞ Norm — Maximum norm : ",np.linalg.norm(v,ord=np.inf),"\n");

x = np.array([[4, 1, 0],
              [2, 3, 7],
              [9, 4, 5]]);

eigvals, eigvecs = np.linalg.eig(x)

print(eigvals,"\n", eigvecs)

