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

def jacobi(A, b, x0, max_iter, tol):
    """
    Solve the system of linear equations Ax = b using the Jacobi method.
    
    Parameters:
    A: a matrix of size (n, n)
    b: a vector of size (n)
    x0: a vector of size (n) containing the initial guess for the solution
    max_iter: the maximum number of iterations to perform
    tol: the tolerance for the solution (the difference between the current and previous solutions)
    
    Returns:
    x: a vector of size (n) containing the solution
    """
    x = x0.copy()  # create a copy of the initial guess
    for i in range(max_iter):  # iterate up to the maximum number of iterations
        x_new = x.copy()  # create a copy of the current solution
        for j in range(len(x)):  # for each element in the solution
            s = sum(A[j][k] * x[k] for k in range(len(x)) if k != j)  # compute the weighted sum of the neighbors
            x_new[j] = (b[j] - s) / A[j][j]  # compute the new value for the element
        if np.linalg.norm(x_new - x) < tol:  # check for convergence
            return i+1, x_new
        x = x_new  # update the solution
    return max_iter, x

A, b = input_MatVec()

#Initial guess x0 for solution x
x0 = np.ones(np.size(b),dtype=float)
its = 0

#Parameters for Jacobi solve
max_iter=100; tol=1e-6

#Solve with Jacobi
print('Solving Problem...')
its, x = jacobi(A, b, x0, max_iter, tol)

print('Required Iterations:', its, '/', max_iter)
print('Solution:', x)

#Calculate Error on Solve
print('Residual Error:',np.sum(np.matmul(A,x)-b))
