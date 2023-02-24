import numpy as np

def input_Mat():
    CorrectMatrix = False
    while True:
        Rows = int(input('Enter n_Rows in Matrix: '))
        Cols = int(input('Enter n_Cols in Matrix: '))
        A = np.zeros((Rows,Cols),dtype=float)
        print('Enter Matrix:')
        for i in range(Rows):
            A[i,:] = list(map(int,input(f"\nEnter Values in Row {i} : ").strip().split()))[:Cols]
        print('Printing Resulting Matrix...')
        for i in range(Rows):
            print(A[i,:])
        
        CorrectMatrix = str(input('Is this Matrix Correct? (Y/N) '))
        if CorrectMatrix == 'y' or CorrectMatrix == 'Y':
            break
        else:
            print('Retrying...')
            print()
        
    return A, Rows

# Gauss Elimination
def Gauss_Elim(A,x):

    Size = np.size(x)
    # Applying Gauss Elimination
    for i in range(Size):
            
        for j in range(i+1, Size):
            ratio = A[j][i]/A[i][i]
            
            for k in range(Size+1):
                A[j][k] = A[j][k] - ratio * A[i][k]

    # Back Substitution
    x[Size-1] = A[Size-1][Size]/A[Size-1][Size-1]

    for i in range(Size-2,-1,-1):
        x[i] = A[i][Size]
        
        for j in range(i+1,Size):
            x[i] = x[i] - A[i][j]*x[j]
        
        x[i] = x[i]/A[i][i]

    return x

A, Rows = input_Mat()

x = np.zeros(Rows,dtype=float)

# Perform Gauss Elim
x = Gauss_Elim(A,x)

# Print Solutiom
print('Solution:', x)
