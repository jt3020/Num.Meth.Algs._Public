import numpy as np 
import matplotlib.pyplot as plt

def Grid_Compress(xi):
    L = 2
    x = (np.exp(L*xi)-1)/(np.exp(L)-1)
    x = x**2
    return(x)

def Grid_Linear(xi,Start,End):
    x = (1-xi)*Start + xi*End
    return(x)

def Grid_Identity(xi):
    x = xi 
    return(x)

def Grid_Tangent(xi):
    x = np.tan(np.pi*xi/4.0)
    return(x)


Steps = int(input("Enter the desired number of steps: "))

xi = np.zeros(Steps+1)
x = np.zeros(Steps+1)

for i in range(Steps+1):
    xi[i] = i*(1/Steps)

print("Mesh Types: 1 - Linear, 2 - Compress, 3 - Identity, 4 - Tangent")
Transform = int(input("Enter the desired 1D mesh type: "))

if Transform == 1:
    x = Grid_Linear(xi,0.0,1.0)
elif Transform == 2:
    x = Grid_Compress(xi)
elif Transform == 3:
    x = Grid_Identity(xi)
elif Transform == 4:
    x = Grid_Tangent(xi)
else:
    print("Unknown Mesh Type - Defaulting to Linear")
    x = Grid_Linear(xi,0.0,1.0)

# Plot mesh
plt.figure()
plt.hlines(1,0,1)
plt.eventplot(x, orientation='horizontal')
plt.show()
