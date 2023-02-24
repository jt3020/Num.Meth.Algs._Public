import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as st
from PIL import ImageTk, Image


dpi=1000


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
def S_find_L(u,xf,yf,zf=None):
    if zf != None: # 3D Problem
        fl = lambda u : np.sqrt(deriv_eval(u,xf)**2+deriv_eval(u,yf)**2+deriv_eval(u,zf)**2)
        L = integrate.quadrature(fl,u[0],u[1],maxiter=2000)[0]
        return L
    else: # 2D Problem
        fl = lambda u : np.sqrt(deriv_eval(u,xf)**2+deriv_eval(u,yf)**2)
        L = integrate.quadrature(fl,u[0],u[1],maxiter=2000)[0]
        return L

# Use Scipy and Gaussuan Quadrature to integrate the function
def SG_integrate(func,u):
    fn = lambda u : func_eval(u,func)
    R = integrate.quadrature(fn,u[0],u[1],maxiter=2000)[0]
    return R

# Calculate the position of the points relevant to the discretisation
def calc_points(N,u0,u1,pf,xf,yf,zf=None):
    x=np.zeros(N)
    y=np.zeros_like(x)
    u_limits=np.array([u0,u1])
    u = np.linspace(u_limits[0],u_limits[1],N)
    s = np.zeros(N)
    if zf != None: # 3D Problem
        z=np.zeros_like(x)
        L = S_find_L(u_limits,xf,yf,zf)
    else: # 2D Problem
        L = S_find_L(u_limits,xf,yf)

    for i in range(N):
        s[i] = ((u[i]-u_limits[0])/(u_limits[1]-u_limits[0]))*L+u[i]*SG_integrate("u*("+pf+")",u_limits)+u[i]*SG_integrate(pf,[u[i],u1])-SG_integrate(pf,[u0,u[i]])


    if zf != None: # 3D Problem
        rf = "np.sqrt(deriv_eval(u,\""+str(xf)+"\")**2"+"+deriv_eval(u,\""+str(yf)+"\")**2"+"+deriv_eval(u,\""+str(zf)+"\")**2)"
    else: # 2D Problem
        rf = "np.sqrt(deriv_eval(u,\""+str(xf)+"\")**2"+"+deriv_eval(u,\""+str(yf)+"\")**2)"
    for i in range(N):
        nrf = str(s[i])+"-SG_integrate('"+str(rf)+"',["+str(u[0])+",u])"
        u[i] = newtonsmethod(nrf,u[i])
        x[i] = func_eval(u[i],xf)
        y[i] = func_eval(u[i],yf)
        if zf != None: # 3D Problem
            z[i] = func_eval(u[i],zf)
    if zf != None: # 3D Problem
        return x,y,z
    else:
        return x,y

def calc_s(N,u0,u1,pf,xf,yf,zf=None):
    u_limits=np.array([u0,u1])
    delta_u = (u_limits[1]-u_limits[0])/(N-1)
    u = np.linspace(u_limits[0],u_limits[1],N)
    Pdisc = np.zeros(N)
    sp = np.zeros(N)
    d2s = np.zeros([N-2,N-2])
    ds = np.zeros([N-2,N-2])
    if zf != None: # 3D Problem
        L = S_find_L(u_limits,xf,yf,zf)
    else: # 2D Problem
        L = S_find_L(u_limits,xf,yf)
    sp_copy = np.zeros_like(sp)
    for i in range(N):
        sp[i] = np.double((i/(N-1)))*L
        Pdisc[i] = func_eval(u[i],pf)
        if i != 0 and i != N-1:
            if i == 1:
                ds[i-1,i] = 1
                d2s[i-1,i-1] = -2
                d2s[i-1,i] = 1
            elif i == N-2:
                ds[i-1,i-2] = -1
                d2s[i-1,i-2] = 1
                d2s[i-1,i-1] = -2
            else:
                ds[i-1,i-2] = -1
                ds[i-1,i] = 1
                d2s[i-1,i-2] = 1
                d2s[i-1,i-1] = -2
                d2s[i-1,i] = 1

    while(any(np.abs(sp_copy - sp)>1e-6)):
        sp_copy = np.copy(sp)
        BCs = np.matmul(ds,sp[1:N-1])
        BCs[0] = BCs[0] - sp[0]
        BCs[-1] = BCs[-1] + sp[-1]
        BCs = (Pdisc[1:N-1]/(8*delta_u))*BCs**3
        BCs[0] = BCs[0] + sp[0]
        BCs[-1] = BCs[-1] + sp[-1]
        sp[1:N-1] = np.matmul(np.linalg.inv(d2s),-BCs)
    

    N = 1000
    delta_u = (u_limits[1]-u_limits[0])/(N-1)
    u = np.linspace(u_limits[0],u_limits[1],N)
    Pdisc = np.zeros(N)
    s = np.zeros(N)
    d2s = np.zeros([N-2,N-2])
    ds = np.zeros([N-2,N-2])
    s_copy = np.zeros_like(s)
    for i in range(N):
        s[i] = np.double((i/(N-1)))*L
        Pdisc[i] = func_eval(u[i],pf)
        if i != 0 and i != N-1:
            if i == 1:
                ds[i-1,i] = 1
                d2s[i-1,i-1] = -2
                d2s[i-1,i] = 1
            elif i == N-2:
                ds[i-1,i-2] = -1
                d2s[i-1,i-2] = 1
                d2s[i-1,i-1] = -2
            else:
                ds[i-1,i-2] = -1
                ds[i-1,i] = 1
                d2s[i-1,i-2] = 1
                d2s[i-1,i-1] = -2
                d2s[i-1,i] = 1
    while(any(np.abs(s_copy - s)>1e-6)):
        s_copy = np.copy(s)
        BCs = np.matmul(ds,s[1:N-1])
        BCs[0] = BCs[0] - s[0]
        BCs[-1] = BCs[-1] + s[-1]
        BCs = (Pdisc[1:N-1]/(8*delta_u))*BCs**3
        BCs[0] = BCs[0] + s[0]
        BCs[-1] = BCs[-1] + s[-1]
        s[1:N-1] = np.matmul(np.linalg.inv(d2s),-BCs)
    return s, sp

def gen_curve(u0,u1,xf,yf,zf=None):
    u=np.linspace(u0,u1,1000)
    if zf != None:
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
    else:
        x=np.zeros(1000)
        y=np.zeros_like(x)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.plot(x,y)
        plt.savefig("curve.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)


def disc_curve(N,u0,u1,pf,xf,yf,zf=None):
    u=np.linspace(u0,u1,1000)
    if zf != None:
        x=np.zeros(1000)
        y=np.zeros_like(x)
        z=np.zeros_like(x)
        xp=np.zeros(N)
        yp=np.zeros_like(xp)
        zp=np.zeros_like(xp)
        xp,yp,zp=calc_points(N,u0,u1,pf,xf,yf,zf)
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
    else:
        x=np.zeros(1000)
        y=np.zeros_like(x)
        xp=np.zeros(N)
        yp=np.zeros_like(xp)
        xp,yp=calc_points(N,u0,u1,pf,xf,yf)
        for i in range(1000):
            x[i]=func_eval(u[i],xf)
            y[i]=func_eval(u[i],yf)
        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.axis("equal")
        plt.minorticks_on()
        plt.grid(which='major',linestyle='-',color='gray')
        plt.grid(which='minor',linestyle=':',color='gray')
        plt.plot(x,y,'b-')
        plt.plot(xp,yp,'ko--')
        plt.plot(xp[0],yp[0],'go')
        plt.plot(xp[-1],yp[-1],'ro')
        plt.savefig("disc_curve.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

def dist_func(N,u0,u1,pf,xf,yf,zf=None):
    u=np.linspace(u0,u1,1000)
    up=np.linspace(u0,u1,N)
    s=np.zeros_like(u)
    if zf != None:
        s,sp=calc_s(N,u0,u1,pf,xf,yf,zf)
    else:
        s,sp=calc_s(N,u0,u1,pf,xf,yf)
    plt.figure(figsize=(6000/dpi,6000/dpi))
    plt.plot(u,s,'b-')
    plt.plot(up,sp,'ko')
    for i in range(N):
        plt.hlines(sp[i],up[0],up[i],linestyles='--',colors='k')
        plt.vlines(up[i],sp[0],sp[i],linestyles='--',colors='k')
    plt.xlim(np.min(u),np.max(u))
    plt.ylim(np.min(s),np.max(s))
    plt.minorticks_on()
    plt.grid(which='major',linestyle='-',color='gray')
    plt.grid(which='minor',linestyle=':',color='gray')
    plt.savefig("dist_func.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

class Window:
    def __init__(self,instance):
        lbl=tk.Label(instance, text="Curve Discretisation", fg='red', font=("Helvetica", 20),anchor="w")
        lbl.place(x=200, y=50)
        self.img = ImageTk.PhotoImage(Image.open("hold.png"))

        self.g_curve=tk.Button(instance, text='Generate curve',fg='red',command=self.gen_curve)
        self.g_curve.place(x=700,y=700)
        self.d_curve=tk.Button(instance, text='Discretise curve',fg='red',command=self.disc_curve)
        self.d_curve.place(x=700,y=725)
        self.d_curve=tk.Button(instance, text='Distance Function',fg='red',command=self.dist_func)
        self.d_curve.place(x=700,y=750)

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

        self.p_lbl=tk.Label(instance, text="p(u)", fg='black', font=("Helvetica", 10),anchor="w")
        self.p_lbl.place(x=25,y=300)
        self.p=tk.Entry(instance)
        self.p.place(x=100,y=300)

        self.N_lbl=tk.Label(instance, text="Nodes",fg='black', font=("Helvetica", 10),anchor="w")
        self.N_lbl.place(x=25,y=350)
        self.N=tk.Entry(instance)
        self.N.place(x=100,y=350)

        self.u0_lbl=tk.Label(instance, text="u0", fg='black', font=("Helvetica", 10),anchor="w")
        self.u0_lbl.place(x=25,y=400)
        self.u0=tk.Entry(instance,width=5)
        self.u0.place(x=50,y=400)

        self.u1_lbl=tk.Label(instance, text="u1", fg='black', font=("Helvetica", 10),anchor="w")
        self.u1_lbl.place(x=100,y=400)
        self.u1=tk.Entry(instance,width=5)
        self.u1.place(x=125,y=400)

        self.out=st.ScrolledText(instance, height = 8)
        self.out.place(x=100, y=700)

        self.imgframe=tk.Frame(instance,width=600,height=600,highlightbackground="black",highlightthickness=1)
        self.imgframe.place(x=300,y=100)
        self.display = tk.Label(self.imgframe,image=self.img)
        self.display.pack()

    def gen_curve(self):
        self.out.insert(tk.END,"Generating Curve\n")
        if int(self.dim.get()) == 2:
            gen_curve(np.double(self.u0.get()),np.double(self.u1.get()),str(self.x.get()),str(self.y.get()),zf=None)
        elif int(self.dim.get()) == 3:
            gen_curve(np.double(self.u0.get()),np.double(self.u1.get()),str(self.x.get()),str(self.y.get()),str(self.z.get()))
        self.img = ImageTk.PhotoImage(Image.open("curve.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()

    def disc_curve(self):
        self.out.insert(tk.END,"Discretising Curve\n")
        if int(self.dim.get()) == 2:
            disc_curve(int(self.N.get()),np.double(self.u0.get()),np.double(self.u1.get()),str(self.p.get()),str(self.x.get()),str(self.y.get()),zf=None)
        elif int(self.dim.get()) == 3:
            disc_curve(int(self.N.get()),np.double(self.u0.get()),np.double(self.u1.get()),str(self.p.get()),str(self.x.get()),str(self.y.get()),str(self.z.get()))
        self.img = ImageTk.PhotoImage(Image.open("disc_curve.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()

    def dist_func(self):
        self.out.insert(tk.END,"Generating Distance Function\n")
        if int(self.dim.get()) == 2:
            dist_func(int(self.N.get()),np.double(self.u0.get()),np.double(self.u1.get()),str(self.p.get()),str(self.x.get()),str(self.y.get()),zf=None)
        elif int(self.dim.get()) == 3:
            dist_func(int(self.N.get()),np.double(self.u0.get()),np.double(self.u1.get()),str(self.p.get()),str(self.x.get()),str(self.y.get()),str(self.z.get()))
        self.img = ImageTk.PhotoImage(Image.open("dist_func.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()


window = tk.Tk()
mywin=Window(window)
window.title('TFI')
window.geometry("1000x800+100+100")
window.mainloop()