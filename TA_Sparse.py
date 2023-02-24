import numpy as np

#Thomas Algorithm
def T_alg_Sparse(A_u,A_d,A_l,b,x):
    """Solves a tridiagonal system using the Thomas Algorithm"""
    N = np.size(b)
    for i in range(1,N):
        w = A_l[i] / A_d[i-1]
        A_d[i] = A_d[i] - A_u[i-1] * w
        A_l[i] = A_l[i] - A_d[i-1] * w
        b[i] = b[i] - b[i-1] * w

    for i in range(N-1,-1,-1):
        if i == N-1:
            x[i] = b[i] / A_d[i]
        else:
            x[i] = (b[i] - A_u[i] * x[i+1]) / A_d[i]

    return x

## Note: Matrix MUST be symmetric and Tridiagonal
## Edit the matrix A below:
A_u = np.array(   [2, 4, 0] ,dtype=float)
A_d = np.array(   [2, 3, 5] ,dtype=float)
A_l = np.array(   [0, 2, 4] ,dtype=float)
# Edit the matrix B below
b = np.array(   [4, 2, 1] ,dtype=float)

x= np.zeros_like(b)

x = T_alg_Sparse(A_u,A_d,A_l,b,x)

print('Solution:', x)
