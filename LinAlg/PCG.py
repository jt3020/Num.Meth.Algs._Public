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

# Pre-Conditioned Conjugate Gradient Method
def PCG_Solver(A,M,b,x,tolerance):
    r = b - np.dot(A,x)
    z = np.dot(M,r)
    p = z
    rzold = np.dot(r,z)
    k = 0
    while np.linalg.norm(r) > tolerance:
        Ap = np.dot(A,p)
        alpha = np.dot(r,z)/np.dot(p,Ap)
        x = x + np.dot(alpha,p)
        r = r - np.dot(alpha,Ap)
        z = np.dot(M,r)
        rznew = np.dot(r,z)
        beta = rznew/rzold
        p = z + np.dot((rznew/rzold),p)
        rzold = rznew
        k = k + 1
    
    return np.array(x), k

def Generate_Pre(A,P):
    # Jacobi
    def Jacobi(A,P):
        n = len(A)
        P = np.zeros([n,n])

        for i in range(n):  
            P[i,i] = 1/A[i,i]
        return P

    # SSOR
    def SSOR(A,P):
        n = len(A)
        D = np.zeros([n,n],dtype=float)
        L = np.zeros([n,n],dtype=float)

        for i in range(n):
            for j in range(n):
                if i == j:
                    D[i,j] = A[i,j]
                elif i > j:
                    L[i,j] = A[i,j]
        P = np.transpose(D+L)
        P = np.matmul(np.linalg.inv(D),P)
        P = np.matmul(D+L,P)
        return P

    # ILU
    def ILU(A,P):
        n = len(A)
        P = np.zeros([n,n],dtype=float)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if (A[i,k] != 0.0) and (A[k,j] != 0.0):
                        P[i,j] = P[i,j] - (A[i,k]*A[k,j]/A[k,k])

        return P

    print()
    print('Enter Desired Preconditioner: 1) - Jacobi, 2) - SSOR, 3) - ILU')
    Pre_Type = int(input('Desired Preconditioner: '))
    if Pre_Type == 1:
        P = Jacobi(A,P)
    elif Pre_Type == 2:
        P = SSOR(A,P)
    elif Pre_Type == 3:
        P = ILU(A,P)
    else:
        print('ERROR: Unknown Preconditioner ID specified')
        exit()

    return P
    

A, b = input_MatVec()
# A = np.array(  [[1., 2., 0.,],
#                 [2., 4., 5.,],
#                 [0., 5., 6.]], dtype=float)
# b = np.array(   [1., 2., 3.], dtype=float)
x = np.ones(np.size(b),dtype=float)
M = np.zeros([np.size(b),np.size(b)],dtype=float)

M = Generate_Pre(A,M)

x, its = PCG_Solver(A,M,b,x,1e-5)

print('Solution:',x)
print('Required Iterations:', its)

#Calculate Error on Solve
print('Residual Error:',np.sum(np.matmul(A,x)-b))
