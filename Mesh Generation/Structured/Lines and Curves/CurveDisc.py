import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.optimize
from scipy.optimize import fsolve
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as st
from PIL import ImageTk, Image


####################################################
# Curve Discretisation Program
# Created for the ME4 Advanced Numerical Methods Course
#
# Utilises a GUI to allow users to generate and discretise curves
# in 2 or 3 dimensions that have been defined parametrically in 'u'
#
# e.g. the curve y = x^2 becomes:
# y = u**2
# x = u
#
# Discretisation method for the curves can also be optionally set
# Otherwise defaults to 'Linear'
# 'Exponential' and 'Tangent' are preset delta spacings for the line
#
# A custom discretisation can also be set parametrically in u
# Note that f(0) = 0 and f(1) = 1 must be satisified for this to work
#
# e.g. For linear: u
# e.g. For a quadratic function: u**2
####################################################

dpi=1000

# Newton's method - finds root of fn with f(x)
def newtonsmethod(func,init_guess):
    x=init_guess
    error = 0.0001
    def f(u):
        f = func_eval(u,func)
        return f

    def df(u):
        df = deriv_eval(u,func)
        return df

    i = x - (f(x)/df(x))
    x = i
    while (np.abs(f(x)) > error):
        i = x - (f(x)/df(x))
        x = i
    return x 

# Use Scipy to calculate the value of the arc length L
def S_find_L(width,xf,yf,zf=None):
    width = 1
    if zf != None: # 3D Problem
        fl = lambda u : np.sqrt(deriv_eval(u,xf)**2+deriv_eval(u,yf)**2+deriv_eval(u,zf)**2)
        L = integrate.quadrature(fl,0,width,maxiter=200)[0]
        return L
    else: # 2D Problem
        fl = lambda u : np.sqrt(deriv_eval(u,xf)**2+deriv_eval(u,yf)**2)
        L = integrate.quadrature(fl,0,width,maxiter=200)[0]
        return L

# Use Scipy and Gaussuan Quadrature to integrate the function
def SG_integrate(func,width):
    fn = lambda u : func_eval(u,func)
    R = integrate.quadrature(fn,0,width,maxiter=200)[0]
    return R

# Calculate the position of the points relevant to the discretisation
def calc_points(N,disc,xf,yf,zf=None):
    x=np.zeros(N)
    y=np.zeros_like(x)
    u=1.0
    if zf != None: # 3D Problem
        z=np.zeros_like(x)
        L = S_find_L(1,xf,yf,zf)
        rf = "np.sqrt(deriv_eval(u,\""+str(xf)+"\")**2"+"+deriv_eval(u,\""+str(yf)+"\")**2"+"+deriv_eval(u,\""+str(zf)+"\")**2)"
        for i in range(0,N):
            # Use desired disc function
            if disc == 'Linear' or disc == '':
                val = (i/(N-1))
            elif disc == 'Exponential':
                val = ((np.exp(L*(i/(N-1)))-1)/(np.exp(L)-1))**2
            elif disc == 'Tangent':
                val = np.tan(np.pi*(i/(N-1))/4.0)
            else:
                val = func_eval(float(i/(N-1)),disc)
            # Find points with newton's method
            nrf = nrf = str(val)+"*"+str(L)+"-SG_integrate('"+str(rf)+"',u)"
            u = newtonsmethod(nrf,i/(N-1))
            x[i] = func_eval(u,xf)
            y[i] = func_eval(u,yf)
            z[i] = func_eval(u,zf)
        return x, y, z
    else: # 2D Problem
        L = S_find_L(1,xf,yf)
        rf = "np.sqrt(deriv_eval(u,\""+str(xf)+"\")**2"+"+deriv_eval(u,\""+str(yf)+"\")**2)"
        for i in range(0,N):
            # Use desired disc function
            if disc == 'Linear' or disc == '':
                val = (i/(N-1))
            elif disc == 'Exponential':
                val = ((np.exp(L*(i/(N-1)))-1)/(np.exp(L)-1))
            elif disc == 'Tangent':
                val = np.tan(np.pi*(i/(N-1))/4.0)
            else:
                val = func_eval(float(i/(N-1)),disc)
            # Find points with newton's method
            nrf = nrf = str(val)+"*"+str(L)+"-SG_integrate('"+str(rf)+"',u)"
            u = newtonsmethod(nrf,i/(N-1))
            x[i] = func_eval(u,xf)
            y[i] = func_eval(u,yf)
        return x, y

# Evaluate the function at u
def func_eval(u,func):
    def f(u):
        f = eval(func)
        return f
    s = f(u)
    return s

# Evaluate the derivative of the curve at u through central difference
def deriv_eval(u,func):
    def f(u):
        f = func_eval(u,func)
        return f
    delta = 1e-6
    forward = f(u+delta)
    backward = f(u-delta)
    s = (forward-backward)/(2*delta)
    return s

# Generate the curve according to parametrically defined functions
def gen_curve(xf,yf,zf=None):
    u=np.linspace(0,1,1000)
    if zf != None: # 3D Problem
        x=np.zeros(1000)
        y=np.zeros_like(x)
        z=np.zeros_like(x)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
            z[i]=func_eval(u[i],zf)
        fig = plt.figure(figsize=(6000/dpi,6000/dpi))
        ax = fig.add_subplot(projection='3d')
        ax.plot(x,y,z)
        fig.savefig("curve.jpg")
    else: # 2D Problem
        x=np.zeros(1000)
        y=np.zeros_like(x)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.plot(x,y)
        plt.savefig("curve.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

# Discretise the curve according to the parametrically defined
# functions, desired node number and discretisation function
def disc_curve(N,disc,xf,yf,zf=None):
    u=np.linspace(0,1,1000)
    if zf != None: # 3D Problem
        x=np.zeros(1000)
        y=np.zeros_like(x)
        z=np.zeros_like(x)
        xp=np.zeros(N)
        yp=np.zeros_like(xp)
        zp=np.zeros_like(xp)
        xp,yp,zp=calc_points(N,disc,xf,yf,zf)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
            z[i]=func_eval(u[i],zf)
        fig = plt.figure(figsize=(6000/dpi,6000/dpi))
        ax = fig.add_subplot(projection='3d')
        ax.plot(x,y,z,'b-')
        ax.plot(xp,yp,zp,'ko--')
        ax.plot(xp[0],yp[0],zp[0],'go')
        ax.plot(xp[-1],yp[-1],zp[-1],'ro')
        fig.savefig("disc_curve.jpg")
    else: # 2D Problem
        x=np.zeros(1000)
        y=np.zeros_like(x)
        xp=np.zeros(N)
        yp=np.zeros_like(xp)
        xp,yp=calc_points(N,disc,xf,yf)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.axis("equal")
        plt.plot(x,y,'b-')
        plt.plot(xp,yp,'ko--')
        plt.plot(xp[0],yp[0],'go')
        plt.plot(xp[-1],yp[-1],'ro')
        plt.savefig("disc_curve.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

# Create the GUI and pass user inputs to relevant functions
class Window:
    def __init__(self,instance):
        lbl=tk.Label(instance, text="Curve Discretisation", fg='red', font=("Helvetica", 20),anchor="w")
        lbl.place(x=200, y=50)
        self.img = ImageTk.PhotoImage(Image.open("hold.png"))

        self.g_curve=tk.Button(instance, text='Generate curve',fg='red',command=self.gen_curve)
        self.g_curve.place(x=700,y=700)
        self.d_curve=tk.Button(instance, text='Discretise curve',fg='red',command=self.disc_curve)
        self.d_curve.place(x=700,y=725)

        self.dim_lbl=tk.Label(instance, text="Dimensions", fg='black', font=("Helvetica", 10),anchor="w")
        self.dim_lbl.place(x=25,y=100)
        self.dim=ttk.Combobox(instance, values=("2","3"))
        self.dim.place(x=100, y=100)

        self.x_lbl=tk.Label(instance, text="x(u)", fg='black', font=("Helvetica", 10),anchor="w")
        self.x_lbl.place(x=25,y=150)
        self.x=tk.Entry(instance)
        self.x.place(x=100,y=150)


        self.y_lbl=tk.Label(instance, text="y(u)", fg='black', font=("Helvetica", 10),anchor="w")
        self.y_lbl.place(x=25,y=200)
        self.y=tk.Entry(instance)
        self.y.place(x=100,y=200)


        self.z_lbl=tk.Label(instance, text="z(u)", fg='black', font=("Helvetica", 10),anchor="w")
        self.z_lbl.place(x=25,y=250)
        self.z=tk.Entry(instance)
        self.z.place(x=100,y=250)

        self.N_lbl=tk.Label(instance, text="Nodes", fg='black', font=("Helvetica", 10),anchor="w")
        self.N_lbl.place(x=25,y=300)
        self.N=tk.Entry(instance)
        self.N.place(x=100,y=300)

        self.disc_lbl=tk.Label(instance, text="Disc", fg='black', font=("Helvetica", 10),anchor="w")
        self.disc_lbl.place(x=25,y=350)
        self.disc=ttk.Combobox(instance, values=("Linear","Exponential","Tangent"))
        self.disc.place(x=100,y=350)

        self.out=st.ScrolledText(instance, height = 8)
        self.out.place(x=100, y=700)

        self.imgframe=tk.Frame(instance,width=600,height=600,highlightbackground="black",highlightthickness=1)
        self.imgframe.place(x=300,y=100)
        self.display = tk.Label(self.imgframe,image=self.img)
        self.display.pack()

    def gen_curve(self):
        self.out.insert(tk.END,"Generating Curve\n")
        if int(self.dim.get()) == 2:
            gen_curve(str(self.x.get()),str(self.y.get()))
        elif int(self.dim.get()) == 3:
            gen_curve(str(self.x.get()),str(self.y.get()),str(self.z.get()))
        self.img = ImageTk.PhotoImage(Image.open("curve.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()

    def disc_curve(self):
        self.out.insert(tk.END,"Discretising Curve\n")
        if int(self.dim.get()) == 2:
            disc_curve(int(self.N.get()),str(self.disc.get()),str(self.x.get()),str(self.y.get()))
        elif int(self.dim.get()) == 3:
            disc_curve(int(self.N.get()),str(self.disc.get()),str(self.x.get()),str(self.y.get()),str(self.z.get()))
        self.img = ImageTk.PhotoImage(Image.open("disc_curve.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()


window = tk.Tk()
mywin=Window(window)
window.title('Curve Discretisation')
window.geometry("1000x800+100+100")
window.mainloop()



