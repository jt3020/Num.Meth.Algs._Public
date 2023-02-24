import math 
import numpy as np 
import matplotlib.pyplot as plt

def Grid_Parabolic(xi,eta):
    r = 1 + xi 
    s = 1 + eta 
    x = (r**2 - s**2) / 2.0
    y = r * s
    return(x,y)

def Grid_Identity(xi,eta):
    x = xi 
    y = eta 
    return(x,y)

def Grid_Elliptic(xi,eta):
    a = 2
    r = 1 + xi
    s = np.pi * eta 
    x = a * np.cosh(r) * np.cos(s) 
    y = a * np.sinh(r) * np.sin(s)
    return(x,y)

def Grid_Horseshoe(xi,eta):
    rho = 2.0
    b0 =  1.0
    b1 =  2.0
    r = b0 + (b1 - b0) * eta 
    theta = np.pi * ( 1.0 - 2.0 * xi ) / 2.0
    x = rho * r * np.cos(theta) 
    y = r * np.sin(theta) 
    return(x,y)

Steps = int(input("Enter the desired number of steps: "))

xi = np.zeros((Steps+1,Steps+1))
eta = np.zeros((Steps+1,Steps+1))
x = np.zeros((Steps+1,Steps+1))
y = np.zeros((Steps+1,Steps+1))

for i in range(0,Steps+1):
    for j in range(0,Steps+1):
        xi[i][j] = i*(1/Steps)
        eta[i][j] = j*(1/Steps)

print("Mesh Types: 1 - Identity, 2 - Parabolic, 3 - Elliptic, 4 - Horseshoe")
Transform = int(input("Enter the desired 1D mesh type: "))

if Transform == 1:
    #Identity
    for i in range(0,Steps+1):
        for j in range(0,Steps+1):      
            x[i][j],y[i][j] = Grid_Identity(xi[i][j],eta[i][j])
            plt.scatter(x[i][j],y[i][j],color='black',marker='x')
    plt.title('Identity')
elif Transform == 2:
    #Parabolic
    for i in range(0,Steps+1):
        for j in range(0,Steps+1):      
            x[i][j],y[i][j] = Grid_Parabolic(xi[i][j],eta[i][j])
            plt.scatter(x[i][j],y[i][j],color='black',marker='x')
    plt.title('Parabolic')
elif Transform == 3:
    #Elliptic
    for i in range(0,Steps+1):
        for j in range(0,Steps+1):      
            x[i][j],y[i][j] = Grid_Elliptic(xi[i][j],eta[i][j])
            plt.scatter(x[i][j],y[i][j],color='black',marker='x')
    plt.title('Elliptic')
elif Transform == 4:
    #Horseshoe
    for i in range(0,Steps+1):
        for j in range(0,Steps+1):      
            x[i][j],y[i][j] = Grid_Horseshoe(xi[i][j],eta[i][j])
            plt.scatter(x[i][j],y[i][j],color='black',marker='x')
    plt.title('Horseshoe')
else:
    print("Unknown Mesh Type - Defaulting to Identity")
    #Identity
    for i in range(0,Steps+1):
        for j in range(0,Steps+1):      
            x[i][j],y[i][j] = Grid_Identity(xi[i][j],eta[i][j])
            plt.scatter(x[i][j],y[i][j],color='black',marker='x')
    plt.title('Identity')

plt.xlabel('x')
plt.ylabel('y')
plt.show()
