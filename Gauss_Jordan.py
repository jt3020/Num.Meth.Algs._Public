import numpy as np

# Gauss Jordan
def Gauss_Jordan(A,x):

    Size = np.size(x)
    # Applying Gauss Jordan Elimination
    for i in range(Size):
        for j in range(Size):
            if i != j:
                ratio = A[j][i]/A[i][i]

                for k in range(Size+1):
                    A[j][k] = A[j][k] - ratio * A[i][k]

    for i in range(Size):
        x[i] = A[i][Size]/A[i][i]

    return x

# Enter input matrix
A = np.array([  [1, -1, 1, 3] ,
                [2, 1, 8, 18] ,
                [4, 2, -3, -2]],dtype = float)

Size = np.size(A,1)
x = np.zeros(3,dtype=float)

# Perform Gauss Jordan
x = Gauss_Jordan(A,x)

# Print Solutiom
print('Solution:', x)
