import numpy as np

def Backward_Euler(func,y0,t0,tf,tolerance,delta):
    """Calculate the solution of func(y,t) at tf with initial condition y0 at t0"""
    delta_t = tf - t0
    yf = y0
    residual = yf - y0 - delta_t * func(yf,tf)
    while np.absolute(residual) > tolerance:
        
        m = ((yf + delta - y0 - delta_t * func(yf + delta,tf)) - (yf - y0 - delta_t * func(yf,tf))) / delta
        yf = yf - (yf - y0 - delta_t * func(yf,tf)) / m
        residual = yf - y0 - delta_t * func(yf,tf)
    return yf, tf

def Crank_Nicolson(func,y0,t0,tf,tolerance,delta):
    """Calculate the solution of func(y,t) at tf with initial condition y0 at t0"""
    delta_t = tf - t0
    yf = y0
    residual = yf - y0 - 0.5 * delta_t * (func(y0,t0) + func(yf,tf))
    while np.absolute(residual) > tolerance:
        
        m = ((yf + delta - y0 - 0.5 * delta_t * (func(y0,t0) + func(yf + delta,tf))) - (yf - y0 - 0.5 * delta_t * (func(y0,t0) + func(yf,tf)))) / delta
        yf = yf - (yf - y0 - 0.5 * delta_t * (func(y0,t0) + func(yf,tf))) / m
        residual = yf - y0 - 0.5 * delta_t * (func(y0,t0) + func(yf,tf))
    return yf, tf

def RK45(func,y0,t0,tf,tolerance):
    """Calculate the solution of func(y,t) at tf with initial condition y0 at t0"""
    delta_t = tf - t0
    a = np.array([[0,0,0,0,0,0],
                  [1/4,0,0,0,0,0],
                  [3/32,9/32,0,0,0,0],
                  [1932/2197,-7200/2197,7296/2197,0,0,0],
                  [439/216,-8,3680/513,-845/4104,0,0],
                  [-8/27,2,-3544/2565,1859/4104,-11/40,0]])
    b5 = np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55])
    c = np.array([0,1/4,3/8,12/13,1,1/2])
    d = np.array([1/360,0,-128/4275,-2197/75240,1/50,2/55])
    k = np.zeros_like(c)
    for i in range(6):
        if i == 0:
            k[i] = delta_t * func(y0,t0)
        else:
            k[i] = delta_t * func(y0 + (np.sum(a[i,:] * k[:])),t0 + c[i] * delta_t)
    TE = np.abs(np.sum(k * d))
    if tolerance <= TE:
        delta_t = 0.9 * delta_t * (tolerance / TE) ** (1/5)
        tf = t0 + delta_t
        for i in range(6):
            if i == 0:
                k[i] = delta_t * func(y0,t0)
            else:
                k[i] = delta_t * func(y0 + (np.sum(a[i,:] * k[:])),t0 + c[i] * delta_t)
    yf = y0 + np.sum(b5 * k)
    return yf, tf

def A_B_M(func,y0,t0,tf):
    """Calculate the solution of func(y,t) at tf with initial condition y0 at t0"""
    delta_t = tf - t0
    t1 = t0 + delta_t / 4
    t2 = t0 + 2 * delta_t / 4
    t3 = t0 + 3 * delta_t / 4
    f0 = func(y0,t0)
    y1 = Crank_Nicolson(func,y0,t0,t1,0.0001,0.0001)[0]
    f1 = func(y1,t1)
    y2 = Crank_Nicolson(func,y1,t1,t2,0.0001,0.0001)[0]
    f2 = func(y2,t2)
    y3 = Crank_Nicolson(func,y2,t2,t3,0.0001,0.0001)[0]
    f3 = func(y3,t3)
    temp_x = y3 + (1 / 24) * (delta_t / 4) * (-9 * f0 + 37 * f1 - 59 * f2 + 55 * f3)
    temp_f = func(temp_x,tf)
    yf = y3 + (1 / 24) * (delta_t / 4) * (f1 - 5 * f2 + 19 * f3 + 9 * temp_f)
    return yf, tf
    
 