import numpy as np

def input_MatVec():
    CorrectMatrix = False
    while True:
        Size = int(input('Enter Size of Matrix/Vector: '))
        A = np.zeros((Size,Size),dtype=float)
        print('Enter Matrix:')
        for i in range(Size):
            A[i,:] = list(map(int,input(f"\nEnter Values in Row {i} : ").strip().split()))[:Size]
        print('Printing Resulting Matrix...')
        for i in range(Size):
            print(A[i,:])
        
        CorrectMatrix = str(input('Is this Matrix Correct? (Y/N) '))
        if CorrectMatrix == 'y' or CorrectMatrix == 'Y':
            break
        else:
            print('Retrying...')
            print()

    print()
    CorrectMatrix = False
    while True:
        b = np.zeros((Size),dtype=float)
        print('Enter Vector:')
        b[:] = list(map(int,input(f"\nEnter Values : ").strip().split()))[:Size]
        print('Printing Resulting Vector...')
        print(b[:])
        
        CorrectMatrix = str(input('Is this Vector Correct? (Y/N) '))
        if CorrectMatrix == 'y' or CorrectMatrix == 'Y':
            break
        else:
            print('Retrying...')
            print()
        
    return A, b

def sor(A, b, x0, w, tolerance, max_iterations):
    """
    A: coefficient matrix
    b: right-hand side vector
    x0: initial guess of the solution
    w: relaxation factor
    tolerance: stopping criterion for the relative error
    max_iterations: maximum number of iterations
    """
    x = x0
    iterations = 0
    error = tolerance + 1
    n = len(b)
    
    while error > tolerance and iterations < max_iterations:
        x_new = np.zeros(n)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            x_new[i] = x[i] + w * (x_new[i] - x[i])
        error = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        x = x_new
        iterations += 1
        
    return x, iterations, error



A, b = input_MatVec()
x0 = np.ones(np.size(b),dtype=float)
w = 1.25
tolerance = 1e-6
max_iterations = 1000

print('Solving Problem...')
x, iterations, error = sor(A, b, x0, w, tolerance, max_iterations)
print("Solution: ", x)
print("Number of iterations: ", iterations)
print("SOR error: ", error)

#Calculate Error on Solve
print('Residual Error:',np.sum(np.matmul(A,x)-b))
