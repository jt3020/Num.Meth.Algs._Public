import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as st
from PIL import ImageTk, Image

dpi=1000


# Evaluate the function at u
def ana_eval(t,func):
    def f(t):
        f = eval(func)
        return f
    s = f(t)
    return s

def dydt_eval(y,t,func):
    def f(y,t):
        f = eval(func)
        return f
    s = f(y,t)
    return s

def Forward_Euler(dydt,y0,t0,tf):
    delta_t = tf - t0
    yf = y0 + delta_t * dydt_eval(y0,t0,dydt)
    return yf, tf

def Backward_Euler(dydt,y0,t0,tf,tolerance,delta):
    delta_t = tf - t0
    yf = y0
    res = yf - y0 - delta_t * dydt_eval(yf,tf,dydt)
    while np.abs(res) > tolerance:
        m = ((yf + delta - y0 - delta_t * dydt_eval(yf + delta,tf,dydt))-(yf - y0 - delta_t * dydt_eval(yf,tf,dydt))) / delta
        yf = yf - (yf - y0 - delta_t * dydt_eval(yf,tf,dydt)) / m
        res = yf - y0 - delta_t * dydt_eval(yf,tf,dydt)
    return yf, tf

def Crank_Nicolson(dydt,y0,t0,tf,tolerance,delta):
    delta_t = tf - t0
    yf = y0
    res = yf - y0 - 0.5 * delta_t * (dydt_eval(y0,t0,dydt) + dydt_eval(yf,tf,dydt))
    while np.abs(res) > tolerance:
        m = ((yf + delta - y0 - 0.5 * delta_t * (dydt_eval(y0,t0,dydt)+dydt_eval(yf + delta,tf,dydt)))-(yf - y0 - 0.5 * delta_t * (dydt_eval(y0,t0,dydt) + dydt_eval(yf,tf,dydt)))) / delta
        yf = yf - (yf - y0 - 0.5 * delta_t * (dydt_eval(y0,t0,dydt) + dydt_eval(yf,tf,dydt))) / m
        res = yf - y0 - 0.5 * delta_t * (dydt_eval(y0,t0,dydt) + dydt_eval(yf,tf,dydt))
    return yf, tf

def RK45(dydt,y0,t0,tf,tolerance):
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
            k[i] = delta_t * dydt_eval(y0,t0,dydt)
        else:
            k[i] = delta_t * dydt_eval(y0 + (np.sum(a[i,:] * k[:])),t0 + c[i] * delta_t,dydt)
    TE = np.abs(np.sum(k * d))
    if tolerance <= TE:
        delta_t = 0.9 * delta_t * (tolerance / TE) ** (1/5)
        tf = t0 + delta_t
        for i in range(6):
            if i == 0:
                k[i] = delta_t * dydt_eval(y0,t0,dydt)
            else:
                k[i] = delta_t * dydt_eval(y0 + (np.sum(a[i,:] * k[:])),t0 + c[i] * delta_t,dydt)
    yf = y0 + np.sum(b5 * k)
    return yf, tf

def A_B_M(dydt,y0,t0,tf):
    delta_t = tf - t0
    t1 = t0 + delta_t / 4
    t2 = t0 + 2 * delta_t / 4
    t3 = t0 + 3 * delta_t / 4
    f0 = dydt_eval(y0,t0,dydt)
    y1 = Crank_Nicolson(dydt,y0,t0,t1,0.0001,0.0001)[0]
    f1 = dydt_eval(y1,t1,dydt)
    y2 = Crank_Nicolson(dydt,y1,t1,t2,0.0001,0.0001)[0]
    f2 = dydt_eval(y2,t2,dydt)
    y3 = Crank_Nicolson(dydt,y2,t2,t3,0.0001,0.0001)[0]
    f3 = dydt_eval(y3,t3,dydt)
    temp_x = y3 + (1 / 24) * (delta_t / 4) * (-9 * f0 + 37 * f1 - 59 * f2 + 55 * f3)
    temp_f = dydt_eval(temp_x,tf,dydt)
    yf = y3 + (1 / 24) * (delta_t / 4) * (f1 - 5 * f2 + 19 * f3 + 9 * temp_f)
    return yf, tf

def ode_solve(dydt,meth,y0,t0,tf,t_step,cf,error=None,ana=None):
    if plt.fignum_exists(1):
        if cf == "yes":
            plt.close()
            fig,ax=plt.subplots()
        else:
            fig,ax=plt.gcf(),plt.gca()
    else:
        fig,ax=plt.subplots()
    if ana != None:
        t_ana = np.linspace(t0,tf,1000)
        y_ana = np.zeros_like(t_ana)
        for i in range(1000):
            y_ana[i] = ana_eval(t_ana[i],ana)
        if "analytical" not in ax.get_legend_handles_labels()[1]:
            ax.plot(t_ana,y_ana,label="analytical")
    y = np.array([y0])
    t = np.array([t0])
    while t[-1] < tf:
        if meth == 'Forward Euler':
            y = np.append(y,[Forward_Euler(dydt,y[-1],t[-1],t[-1]+t_step)[0]],axis=0)
            t = np.append(t,[Forward_Euler(dydt,y[-1],t[-1],t[-1]+t_step)[1]],axis=0)
        elif meth == 'Backward Euler':
            y = np.append(y,[Backward_Euler(dydt,y[-1],t[-1],t[-1]+t_step,error,0.0001)[0]],axis=0)
            t = np.append(t,[Backward_Euler(dydt,y[-1],t[-1],t[-1]+t_step,error,0.0001)[1]],axis=0)
        elif meth == 'Crank Nicolson':
            y = np.append(y,[Crank_Nicolson(dydt,y[-1],t[-1],t[-1]+t_step,error,0.0001)[0]],axis=0)
            t = np.append(t,[Crank_Nicolson(dydt,y[-1],t[-1],t[-1]+t_step,error,0.0001)[1]],axis=0)
        elif meth == 'RK45':
            y = np.append(y,[RK45(dydt,y[-1],t[-1],t[-1]+t_step,error)[0]],axis=0)
            t = np.append(t,[RK45(dydt,y[-1],t[-1],t[-1]+t_step,error)[1]],axis=0)
        elif meth == 'ABM':
            y = np.append(y,[A_B_M(dydt,y[-1],t[-1],t[-1]+t_step)[0]],axis=0)
            t = np.append(t,[A_B_M(dydt,y[-1],t[-1],t[-1]+t_step)[1]],axis=0)
    ax.plot(t,y,label=meth+" timestep = {}".format(t_step))
    ax.legend()
    plt.savefig("sol.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

class Window:
    def __init__(self,instance):
        self.idiot_counter = 0

        lbl=tk.Label(instance, text="IVP Solver", fg='red', font=("Helvetica", 20),anchor="w")
        lbl.place(x=200, y=50)
        self.img = ImageTk.PhotoImage(Image.open("hold.png"))

        self.sol=tk.Button(instance, text='Solve',fg='red',command=self.Solve)
        self.sol.place(x=700,y=700)

        self.ana_lbl=tk.Label(instance, text="Analytical\nKnown?", fg='black', font=("Helvetica", 10),anchor="w")
        self.ana_lbl.place(x=15,y=200)
        self.ana=ttk.Combobox(instance, values=("yes","no"))
        self.ana.place(x=125, y=200)

        self.dydt_lbl=tk.Label(instance, text="y'(y,t)", fg='black', font=("Helvetica", 10),anchor="w")
        self.dydt_lbl.place(x=15,y=150)
        self.dydt=tk.Entry(instance)
        self.dydt.place(x=125,y=150)

        self.meth_lbl=tk.Label(instance, text="Method", fg='black', font=("Helvetica", 10),anchor="w")
        self.meth_lbl.place(x=15,y=100)
        self.meth=ttk.Combobox(instance, values=("Forward Euler","Backward Euler","Crank Nicolson","RK45","ABM"))
        self.meth.place(x=125, y=100)


        self.y_lbl=tk.Label(instance, text=" Analytical\nSolution y(t)", fg='black', font=("Helvetica", 10),anchor="w")
        self.y_lbl.place(x=15,y=250)
        self.y=tk.Entry(instance)
        self.y.place(x=125,y=250)

        self.dt_lbl=tk.Label(instance, text="Step", fg='black', font=("Helvetica", 10),anchor="w")
        self.dt_lbl.place(x=15,y=300)
        self.dt=tk.Entry(instance)
        self.dt.place(x=125,y=300)

        self.E_lbl=tk.Label(instance, text="Error\nTolerance", fg='black', font=("Helvetica", 10),anchor="w")
        self.E_lbl.place(x=15,y=350)
        self.E=tk.Entry(instance)
        self.E.place(x=125,y=350)

        self.y0_lbl=tk.Label(instance, text=" y0", fg='black', font=("Helvetica", 10),anchor="w")
        self.y0_lbl.place(x=15,y=400)
        self.y0=tk.Entry(instance)
        self.y0.place(x=125,y=400)

        self.t0_lbl=tk.Label(instance, text=" t0", fg='black', font=("Helvetica", 10),anchor="w")
        self.t0_lbl.place(x=15,y=450)
        self.t0=tk.Entry(instance)
        self.t0.place(x=125,y=450)

        self.tf_lbl=tk.Label(instance, text=" tf", fg='black', font=("Helvetica", 10),anchor="w")
        self.tf_lbl.place(x=15,y=500)
        self.tf=tk.Entry(instance)
        self.tf.place(x=125,y=500)

        self.cf_lbl=tk.Label(instance, text="Clear\nFigure?", fg='black', font=("Helvetica", 10),anchor="w")
        self.cf_lbl.place(x=15,y=550)
        self.cf=ttk.Combobox(instance, values=("yes","no"))
        self.cf.place(x=125, y=550)

        self.out=st.ScrolledText(instance, height = 8)
        self.out.place(x=125, y=700)

        self.imgframe=tk.Frame(instance,width=600,height=600,highlightbackground="black",highlightthickness=1)
        self.imgframe.place(x=300,y=100)
        self.display = tk.Label(self.imgframe,image=self.img)
        self.display.pack()

    def Solve(self):
        self.out.delete("1.0",tk.END)
        if self.idiot_counter > 10:
            self.out.insert(tk.END,"You can't have nice things!\n")
        elif all([str(self.meth.get()) != "Forward Euler", str(self.meth.get()) != "Backward Euler", str(self.meth.get()) != "Crank Nicolson", str(self.meth.get()) != "RK45", str(self.meth.get()) != "ABM"]):
            self.out.insert(tk.END,"Pick a valid solver option.\n")
            self.idiot_counter += 1
        elif all([str(self.ana.get()) != "yes", str(self.ana.get()) != "no"]):
            self.out.insert(tk.END,"Is the analytical solution known, yes or no?\n")
            self.idiot_counter += 1
        elif float(self.dt.get()) <= 0:
            self.out.insert(tk.END,"Invalid step size.\n")
            self.idiot_counter += 1
        elif str(self.ana.get()) == "yes":
            self.out.insert(tk.END,"Solving ODE with "+str(self.meth.get())+"...\n")
            ode_solve(str(self.dydt.get()),str(self.meth.get()),float(self.y0.get()),float(self.t0.get()),float(self.tf.get()),float(self.dt.get()),str(self.cf.get()),error=float(self.E.get()),ana=str(self.y.get()))
        elif str(self.ana.get()) == "no":
            self.out.insert(tk.END,"Solving ODE with "+str(self.meth.get())+"...\n")
            ode_solve(str(self.dydt.get()),str(self.meth.get()),float(self.y0.get()),float(self.t0.get()),float(self.tf.get()),float(self.dt.get()),str(self.cf.get()),error=float(self.E.get()))
        else:
            self.out.insert(tk.END,"Is the analytical function known?\n")
        self.img = ImageTk.PhotoImage(Image.open("sol.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()



window = tk.Tk()
mywin=Window(window)
window.title('IVP')
window.geometry("1000x800+100+100")
window.mainloop()