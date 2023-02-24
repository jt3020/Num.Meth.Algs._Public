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

# Conjugate Gradient Solver
def CG_Solver(A,b,x,tolerance):
    r = b - np.dot(A,x)
    p = r 
    rsold = np.dot(r,r)
    k = 0
    while np.linalg.norm(rsold) > tolerance:
        Ap = np.dot(A,p)
        alpha = rsold/np.dot(p,Ap)
        x = x + np.dot(alpha,p)
        r = r - np.dot(alpha,Ap)
        rsnew = np.dot(r,r)
        p = r + np.dot((rsnew/rsold),p)
        rsold = rsnew
        k = k + 1
    return np.array(x), k

A, b = input_MatVec()
# A = np.array(  [[1., 2., 3.,],
#                 [2., 4., 5.,],
#                 [3., 5., 6.]], dtype=float)
# b = np.array(   [1., 2., 3.], dtype=float)
x = np.ones(np.size(b),dtype=float)
x, its = CG_Solver(A,b,x,1e-5)

print('Solution:',x)
print('Required Iterations:', its)

#Calculate Error on Solve
print('Residual Error:',np.sum(np.matmul(A,x)-b))
