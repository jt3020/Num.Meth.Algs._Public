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

#Thomas Algorithm
def T_alg(A,b,x):
    """Solves a tridiagonal system using the Thomas Algorithm"""
    N = np.size(b)
    for i in range(1,N):
        w = A[i,i-1] / A[i-1,i-1]
        A[i,:] = A[i,:] - A[i-1,:] * w
        b[i] = b[i] - b[i-1] * w

    for i in range(N-1,-1,-1):
        if i == N-1:
            x[i] = b[i] / A[i,i]
        else:
            x[i] = (b[i] - A[i,i+1] * x[i+1]) / A[i,i]

    return x


A, b = input_MatVec()
x= np.zeros_like(b)

#Solve with Thomas Alg
print('Solving Problem...')
x = T_alg(A,b,x)

print('Solution:', x)
#Calculate Error on Solve
print('Residual Error:',np.sum(np.matmul(A,x)-b))
