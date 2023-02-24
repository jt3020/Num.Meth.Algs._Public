import numpy as np
import sys

# Jacobi
def Jacobi(A,P):
    n = len(A)
    P = np.zeros((n,n))

    for i in range(n):  
        P[i,i] = 1/A(i,i)
    return(np.array(P))

# SSOR
def SSOR(A,P):
    n = len(A)
    D = np.zeros((n,n))
    L = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                D[i,j] = A[i,j]
            elif i > j:
                L[i,j] = A[i,j]
    
    P = np.transpose(D+L)
    P = np.matmul(np.invert(D),P)
    P = np.matmul(D+L,P)
    return(np.array(P))

# ILU
def ILU(A,P):
    n = len(A)
    P = np.zeros((n,n))

    for k in range(n-1):
        for i in range(k+1,n):
            if P[i,k] == 0.0:
                P[i,k] = P[i,k]/P[k,k]
                for j in range(k+1,n):
                    if P[i,j] == 0.0:
                        P[i,j] = P[i,j] - P[i,k]*P[k,j]
    return(np.array(P))

n = 5
x = np.ones(n)
b = np.ones(n)
A = np.zeros((n,n))
P = np.zeros((n,n))


