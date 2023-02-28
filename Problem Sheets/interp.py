import numpy as np 
import matplotlib.pyplot as plt

def lagran(n, x, y, c):
    if n <= 0:
        raise ValueError('lagran: n is zero or negative')
    c[0] = y[0]
    if n == 1:
        return
    for k in range(1, n):
        c[k] = y[k]
        km1 = k-1
        for i in range(k):
        # for i in range(km1):
            dif = x[i] - x[k]
            if dif == 0.0:
                raise ValueError('lagran: the abscissas are not distinct')
            c[k] = (c[i] - c[k]) / dif


def evaluate(xx, n, x, c):
    pione = 1.0
    pone = c[0]
    yfit = pone
    if n == 1:
        return
    for k in range(1, n):
        pitwo = (xx - x[k-1]) * pione
        pione = pitwo
        ptwo = pone + pitwo * c[k]
        pone = ptwo
    yfit = ptwo
    return yfit

def spline(ib, n, r, t, aux):
    if n <= 1:
        raise ValueError('spline: n=1 wrong number of points')
    elif n == 2:
        t[0] = r[1] - r[0]
        t[1] = t[0]
    elif n == 3:
        if ib == 1:
            t[1] = 0.25 * (3.0 * (r[2] - r[0]) - t[0] - t[2])
        else:
            t[0] = -1.25 * r[0] + 1.5 * r[1] - 0.25 * r[2]
            t[1] = -0.5 * r[0] + 0.5 * r[2]
            t[2] = 0.25 * r[0] - 1.5 * r[1] + 1.25 * r[2]
    else:
        rv = 3.0 * (r[2] - r[0])
        if ib == 1:
            bet = 4.0
            rv = rv - t[0]
        else:
            bet = 3.5
            rv = rv - 1.5 * (r[1] - r[0])
        t[1] = rv / bet
        # print()
        ## Seems same up to here
        # *** Rows of the type  [ 1,  4,  1 ]

        for j in range(2, n - 3): # Changed 2,n-1 to 2,n-3
            aux[j] = 1.0 / bet
            bet = 4.0 - aux[j]
            if bet == 0.0:
                raise ValueError('spline: zero pivot ?')
            rv = 3.0 * (r[j + 1] - r[j - 1])
            t[j] = (rv - t[j - 1]) / bet
        # print()
        # *** Last row i=n-1   ib=1  -> [ 1, 4] ;  ib = 2  -> [ 1, 3.5 ].

        aux[n - 2] = 1.0 / bet
        rv = 3.0 * (r[n - 1] - r[n - 3])
        if ib == 1:
            bet = 4.0
            rv = rv - t[n - 1]
        else:
            bet = 3.5
            rv = rv - 1.5 * (r[n - 1] - r[n - 2])
        bet = bet - aux[n - 2]
        if bet == 0.0:
            raise ValueError('spline: zero pivot ?')
        t[n - 2] = (rv - t[n - 3]) / bet
        # print()
        # *** Backsubstitution.

        for j in range(n - 3, 0, -1): # Changed n-3,1,-1 to n-3,0,-1
            t[j] = t[j] - aux[j + 1] * t[j + 1]
        # print()
        # *** End values when ib = 2.

        if ib == 2:
            t[0] = 1.5 * (r[1] - r[0]) - 0.5 * t[1]
            t[n - 1] = 1.5 * (r[n - 1] - r[n - 2]) - 0.5 * t[n - 2]
        # print()

def fgcurv(ider, ndim, r1, r2, u, r):
    for id in range(1, ndim+1):
        id1 = id + ndim
        id2 = id1 + ndim
        r12 = r2[id-1] - r1[id-1]
        a1 = r1[id-1]
        a2 = r1[id1-1]
        a3 = 3 * r12 - 2 * r1[id1-1] - r2[id1-1]
        a4 = -2 * r12 + r1[id1-1] + r2[id1-1]
        r[id-1] = a1 + u * (a2 + u * (a3 + u * a4))
        if ider == 0:
            continue
        r[id1-1] = a2 + u * (2 * a3 + 3 * u * a4)
        if ider == 1:
            continue
        r[id2-1] = 2 * a3 + 6 * u * a4




# Relevant arrays to be used later
mxn = 1000
x = [[0] * mxn for _ in range(4)]
v1 = [0] * mxn
vx = [0] * mxn
vy = [0] * mxn
r1 = [0] * mxn
r2 = [0] * mxn
r = [0] * 6

# User input via terminal
print('Functions: 1) - x^(1/3), 2) - 1/(1+x^2)')
fn_ID = int(input('Enter the desired function: '))
n = int(input('Enter the number of points: '))
for i in range(n):
    x[0][i] = float(input('Enter x position: '))
    if fn_ID == 1:
        x[1][i] = x[0][i]**(1/3)
    elif fn_ID == 2:
        x[1][i] = 1/(1+x[0][i]**2)
    else:
        print('ERROR: Unknown Function')

# Perform Lagrange interpolation
for i in range(n):
    vx[i] = x[0][i]
    vy[i] = x[1][i]
lagran(n, vx, vy, v1)

# Evaluate lagrange interpolation and generate curve that can be plotted
m = 21
lagr_curve = np.zeros([m*(n-1),2],dtype=float)
PointID = -1
am = m - 1
for i in range(n-1):
    x1 = vx[i]
    x2 = vx[i+1]
    xi = (x2 - x1) / am
    for j in range(m):
        aj = j
        xx = x1 + aj * xi
        yfit = evaluate(xx, n, vx, v1)
        PointID += 1
        lagr_curve[PointID,:] = xx, yfit

# Perform cubic spline interpolation
for id in range(1, 3):
    for ip in range(1, n+1):
        vx[ip-1] = x[id-1][ip-1]
    spline(2, n, vx, vy, v1)
    for ip in range(1, n+1):
        x[id+1][ip-1] = vy[ip-1]


# Evaluate ferguson cubic spline and generate curve that can be plotted
m = 21
ferg_curve = np.zeros([m*(n-1),2],dtype=float)
am = m - 1
PointID = -1
for i in range(1,n):
    ui = 1. / am
    for j in range(1, m+1):
        aj = j - 1
        u = aj * ui
        ID = -1
        ID2 = -1
        row = -1
        if i == 1:
            while ID < mxn-1:
                row += 1
                for ii in range(len(x)):
                    ID += 1
                    r1[ID] = x[ii][row]
                if row > 0:
                    ID2 += 1
                    r2[ID2] = r1[ID2+4]
        elif j == 1: # Only the first time it loops
            r1[:] = r2[:]
            for k in range(len(r2)-4):
                r2[k] = r2[k+4]
        fgcurv(0,2,r1,r2,u,r)
        PointID += 1
        ferg_curve[PointID,:] = r[0], r[1]

# Plot relevant curves and nodes
plt.plot(lagr_curve[:,0],lagr_curve[:,1],label='Lagrange Interpolation')
plt.plot(ferg_curve[:,0],ferg_curve[:,1],label='Ferguson Spline')
plt.plot(x[0][:n],x[1][:n],'x',label='Nodes')
plt.legend()
plt.show()