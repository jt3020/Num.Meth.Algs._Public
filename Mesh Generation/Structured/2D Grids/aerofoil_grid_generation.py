import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.scrolledtext as st
from PIL import ImageTk, Image



dpi = 1000

def comp_grid(eta,xi):
    if eta <= 0 or xi <= 0:
        return 'Invalid node numbers'
    else:
        
        IMAX = int(eta)
        JMAX = int(xi)

        x = np.zeros([IMAX,JMAX])
        y = np.zeros([IMAX,JMAX])

        for i in range(0,JMAX):
            x[:,i] = np.linspace(0,1,IMAX)
        for j in range(0,IMAX):
            y[j,:] = np.linspace(0,1,JMAX)


        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$\xi$")
        for i in range(0,JMAX):
            plt.plot(x[:,i],y[:,i],'k-')
            plt.plot(x[:,i],y[:,i],'k-')
        for j in range(0,IMAX):
            plt.plot(x[j,:],y[j,:],'k-')
            plt.plot(x[j,:],y[j,:],'k-')

        plt.savefig("comp_grid.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)
        return 'Computational grid generated'

def alg_grid(eta,xi,foil_code,reta,rxi):
    if eta <= 0 or xi <= 0 or foil_code < 1 or foil_code > 99:
        return 'Invalid numbers'
    else:
        IMAX = int(eta)
        JMAX = int(xi)
        thickness = foil_code/100

        x = np.zeros([IMAX,JMAX])
        y = np.zeros([IMAX,JMAX])
        P = np.zeros([IMAX,JMAX])
        Q = np.zeros([IMAX,JMAX])

        x[:,:] = 0.001
        y[:,:] = 0.001

        P[:,:] = 0.
        Q[:,:] = 0.



        dh = 0.0
        r=rxi
        h=3.*(1.-r)/(1.-(r**((IMAX-1)/5)))

        M=IMAX-1
        N=JMAX-1

        for i in range(0,int((IMAX-1)/2)+1):
            if i < (IMAX-1)/5:
                x[i,0] = 4.-dh
                y[i,0] = 0.
                x[i,JMAX-1] = 4-dh
                y[i,JMAX-1] = -2.
                dh = dh + h*(r**i)
            elif i < 7*(IMAX-1)/20:
                x[i,0] = 0.5 + 0.5 * np.cos((0.5*np.pi*(i-((IMAX-1)/5)))/(7*(IMAX-1)/20-(IMAX-1)/5))
                y[i,0] = -5.*thickness*((0.2948*np.sqrt(x[i,0]))-(0.126*x[i,0])-(0.3516*x[i,0]**2)+(0.2843*x[i,0]**3)-(0.1015*x[i,0]**4))
                x[i,JMAX-1] = 1.-3.*((np.sin(((i-((IMAX-1)/5))*np.pi)/(2*(((IMAX-1)/2)-((IMAX-1)/5)))))**1.3)
                y[i,JMAX-1] = -2.*(np.cos((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))
            else:
                x[i,0] = 0.5 - 0.5 * np.sin(0.5*np.pi*(i-(7*(IMAX-1)/20))/(7*(IMAX-1)/20-(IMAX-1)/5))
                y[i,0] = -5*thickness*(0.2948*np.sqrt(x[i,0])-0.126*x[i,0]-0.3516*x[i,0]**2+0.2843*x[i,0]**3-0.1015*x[i,0]**4)
                x[i,JMAX-1] = 1.-3*(np.sin((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))**1.3
                y[i,JMAX-1] = -2*(np.cos((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))

        for i in range(int((IMAX-1)/2)+1,IMAX):
            x[i,0] = x[(IMAX-1)-i,0]
            y[i,0] = -y[(IMAX-1)-i,0]
            x[i,JMAX-1] = x[(IMAX-1)-i,JMAX-1]
            y[i,JMAX-1] = -y[(IMAX-1)-i,JMAX-1]

        dh = 0.0
        r=reta
        h=2.*(1.-r)/(1.-r**(JMAX-1))

        for j in range(0,JMAX):
            x[0,j] = 4.
            y[0,j] = -dh
            x[IMAX-1,j] = 4.
            y[IMAX-1,j] = dh
            dh = dh + h*(r**j)

        for j in range(1,JMAX-1):
            for i in range(1,IMAX-1):
                x[i,j] = (i/M)*x[M,j]+((M-i)/M)*x[0,j]+(j/N)*x[i,N]+((N-j)/N)*x[i,0]-(i/M)*(j/N)*x[M,N]-(i/M)*((N-j)/N)*x[M,0]-((M-i)/M)*(j/N)*x[0,N]-((M-i)/M)*((N-j)/N)*x[0,0]
                y[i,j] = (i/M)*y[M,j]+((M-i)/M)*y[0,j]+(j/N)*y[i,N]+((N-j)/N)*y[i,0]-(i/M)*(j/N)*y[M,N]-(i/M)*((N-j)/N)*y[M,0]-((M-i)/M)*(j/N)*y[0,N]-((M-i)/M)*((N-j)/N)*y[0,0]

        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(0,JMAX):
            plt.plot(x[:,i],y[:,i],'k-')
            plt.plot(x[:,i],y[:,i],'k-')
        for j in range(0,IMAX):
            plt.plot(x[j,:],y[j,:],'k-')
            plt.plot(x[j,:],y[j,:],'k-')

        plt.savefig("alg_grid.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)

        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(x[:,0],y[:,0],'k-')
        plt.plot(x[:,JMAX-1],y[:,JMAX-1],'k-')
        plt.plot(x[0,:],y[0,:],'k-')
        plt.plot(x[IMAX-1,:],y[IMAX-1,:],'k-')

        plt.savefig("outline.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)
        return 'Algebraic grid generated'
    
def elip_grid(eta,xi,foil_code,reta,rxi,omega,a,c,aa,cc,poisson):
    print(eta,xi,foil_code,reta,rxi,omega,a,c,aa,cc,poisson)
    if eta <= 0 or xi <= 0 or foil_code < 1 or foil_code > 99:
        return 'Invalid numbers'
    else:
        IMAX = int(eta)
        JMAX = int(xi)
        thickness = foil_code/100

        x = np.zeros([IMAX,JMAX])
        y = np.zeros([IMAX,JMAX])
        P = np.zeros([IMAX,JMAX])
        Q = np.zeros([IMAX,JMAX])

        x[:,:] = 0.001
        y[:,:] = 0.001

        P[:,:] = 0.
        Q[:,:] = 0.



        dh = 0.0
        r=rxi
        h=3.*(1.-r)/(1.-(r**((IMAX-1)/5)))

        M=IMAX-1
        N=JMAX-1

        for i in range(0,int((IMAX-1)/2)+1):
            if i < (IMAX-1)/5:
                x[i,0] = 4.-dh
                y[i,0] = 0.
                x[i,JMAX-1] = 4-dh
                y[i,JMAX-1] = -2.
                dh = dh + h*(r**i)
            elif i < 7*(IMAX-1)/20:
                x[i,0] = 0.5 + 0.5 * np.cos((0.5*np.pi*(i-((IMAX-1)/5)))/(7*(IMAX-1)/20-(IMAX-1)/5))
                y[i,0] = -5.*thickness*((0.2948*np.sqrt(x[i,0]))-(0.126*x[i,0])-(0.3516*x[i,0]**2)+(0.2843*x[i,0]**3)-(0.1015*x[i,0]**4))
                x[i,JMAX-1] = 1.-3.*((np.sin(((i-((IMAX-1)/5))*np.pi)/(2*(((IMAX-1)/2)-((IMAX-1)/5)))))**1.3)
                y[i,JMAX-1] = -2.*(np.cos((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))
            else:
                x[i,0] = 0.5 - 0.5 * np.sin(0.5*np.pi*(i-(7*(IMAX-1)/20))/(7*(IMAX-1)/20-(IMAX-1)/5))
                y[i,0] = -5*thickness*(0.2948*np.sqrt(x[i,0])-0.126*x[i,0]-0.3516*x[i,0]**2+0.2843*x[i,0]**3-0.1015*x[i,0]**4)
                x[i,JMAX-1] = 1.-3*(np.sin((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))**1.3
                y[i,JMAX-1] = -2*(np.cos((i-(IMAX-1)/5)*np.pi/(2*((IMAX-1)/2-(IMAX-1)/5))))

        dxi = 1./(IMAX-1)
        deta = 1./(JMAX-1)

        for i in range(int((IMAX-1)/2)+1,IMAX):
            x[i,0] = x[(IMAX-1)-i,0]
            y[i,0] = -y[(IMAX-1)-i,0]
            x[i,JMAX-1] = x[(IMAX-1)-i,JMAX-1]
            y[i,JMAX-1] = -y[(IMAX-1)-i,JMAX-1]

        dh = 0.0
        r=reta
        h=2.*(1.-r)/(1.-r**(JMAX-1))

        for j in range(0,JMAX):
            x[0,j] = 4.
            y[0,j] = -dh
            x[IMAX-1,j] = 4.
            y[IMAX-1,j] = dh
            dh = dh + h*(r**j)

        for j in range(1,JMAX-1):
            for i in range(1,IMAX-1):
                x[i,j] = (i/M)*x[M,j]+((M-i)/M)*x[0,j]+(j/N)*x[i,N]+((N-j)/N)*x[i,0]-(i/M)*(j/N)*x[M,N]-(i/M)*((N-j)/N)*x[M,0]-((M-i)/M)*(j/N)*x[0,N]-((M-i)/M)*((N-j)/N)*x[0,0]
                y[i,j] = (i/M)*y[M,j]+((M-i)/M)*y[0,j]+(j/N)*y[i,N]+((N-j)/N)*y[i,0]-(i/M)*(j/N)*y[M,N]-(i/M)*((N-j)/N)*y[M,0]-((M-i)/M)*(j/N)*y[0,N]-((M-i)/M)*((N-j)/N)*y[0,0]

        omega = omega
        res = 1.0
        alpha = 0.
        beta = 0.
        gamma = 0.
        xtemp = 0.
        ytemp = 0.
        ITR = 0

        if poisson == 'yes':
            a=a
            aa=aa
        elif poisson == 'no':
            a = 0
            aa = 0
        c = c
        cc = cc

        PP = 0.
        QQ = 0.

        while(res>0.000001):
            res = 0
            ITR = ITR + 1
            for j in range(1,JMAX-1):
                for i in range(1,IMAX-1):
                    xeta = (x[i,j+1]-x[i,j-1])/(2*deta)
                    yeta = (y[i,j+1]-y[i,j-1])/(2*deta)
                    xxi = (x[i+1,j]-x[i-1,j])/(2*dxi)
                    yxi = (y[i+1,j]-y[i-1,j])/(2*dxi)
                    J = xxi*yeta - xeta*yxi

                    alpha = xeta*xeta+yeta*yeta
                    beta = xxi*xeta + yxi*yeta
                    gamma = xxi*xxi + yxi*yxi

                    if np.abs(np.double(i/(IMAX-1))-0.5) == 0:
                        PP = 0
                    else:
                        PP = -a*(np.double(i/(IMAX-1))-0.5)/(np.abs(np.double(i/(IMAX-1))-0.5))*np.exp(-c*np.abs(np.double(i/(IMAX-1))-0.5))
                    
                    if np.abs(np.double(j/(JMAX-1))-0.0) == 0:
                        QQ = 0
                    else:
                        QQ = -aa*(np.double(j/(JMAX-1))-0.0)/(np.abs(np.double(j/(JMAX-1))-0.0))*np.exp(-cc*np.abs(np.double(j/(JMAX-1))-0.0))

                    xtemp = ((dxi*deta)**2)/(2*(alpha*deta*deta+gamma*dxi*dxi))*\
                            ((alpha/(dxi*dxi))*(x[i+1,j]+x[i-1,j])+(gamma/(deta*deta))*\
                            (x[i,j+1]+x[i,j-1])-(beta/(2.*deta*dxi))*(x[i+1,j+1]+x[i-1,j-1]-x[i-1,j+1]-x[i+1,j-1])\
                            +(J*J)*(xxi*PP+xeta*QQ))

                    ytemp = ((dxi*deta)**2)/(2*(alpha*deta*deta+gamma*dxi*dxi))*\
                            ((alpha/(dxi*dxi))*(y[i+1,j]+y[i-1,j])+(gamma/(deta*deta))*\
                            (y[i,j+1]+y[i,j-1])-(beta/(2*deta*dxi))*(y[i+1,j+1]+y[i-1,j-1]-y[i-1,j+1]-y[i+1,j-1])\
                            +(J*J)*(yxi*PP+yeta*QQ))

                    res = res + (x[i,j]-xtemp)**2 + (y[i,j]-ytemp)**2

                    xtemp = omega*xtemp + (1-omega)*x[i,j]
                    ytemp = omega*ytemp + (1-omega)*y[i,j]

                    x[i,j] = xtemp
                    y[i,j] = ytemp
            res = np.sqrt(res)

        plt.figure(figsize=(6000/dpi,6000/dpi))
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(0,JMAX):
            plt.plot(x[:,i],y[:,i],'k-')
            plt.plot(x[:,i],y[:,i],'k-')
        for j in range(0,IMAX):
            plt.plot(x[j,:],y[j,:],'k-')
            plt.plot(x[j,:],y[j,:],'k-')

        plt.savefig("elip_grid.jpg",dpi=dpi,bbox_inches='tight',pad_inches=0.1)
        return 'Elliptical grid generated'

class Window:
    def __init__(self,instance):
        self.comp=False
        self.alg=False
        self.elip=False
        self.img = ImageTk.PhotoImage(Image.open("hold.png"))
        lbl=tk.Label(instance, text="GUI for Curvilinear Grid Generation of NACA airfoils", fg='red', font=("Helvetica", 20),anchor="w")
        lbl.place(x=200, y=50)

        self.c_grid=tk.Button(instance, text='Generate Computational Grid',fg='red',command=self.comp_grid)
        self.c_grid.place(x=700,y=700)
        self.a_grid=tk.Button(instance, text='Generate Algebraic Grid',fg='red',command=self.alg_grid)
        self.a_grid.place(x=700,y=725)
        self.e_grid=tk.Button(instance, text='Generate Eliptical Grid',fg='red',command=self.elip_grid)
        self.e_grid.place(x=700,y=750)

        self.eta_lbl=tk.Label(instance, text="Eta nodes", fg='black', font=("Helvetica", 10),anchor="w")
        self.eta_lbl.place(x=25,y=100)
        self.eta=tk.Entry(instance)
        self.eta.place(x=100,y=100)

        self.xi_lbl=tk.Label(instance, text="Xi nodes", fg='black', font=("Helvetica", 10),anchor="w")
        self.xi_lbl.place(x=25,y=150)
        self.xi=tk.Entry(instance)
        self.xi.place(x=100,y=150)

        self.fc_lbl=tk.Label(instance, text="NACA code", fg='black', font=("Helvetica", 10),anchor="w")
        self.fc_lbl.place(x=25,y=200)
        self.fc=ttk.Entry(instance)
        self.fc.place(x=100, y=200)

        self.reta_lbl=tk.Label(instance, text="Eta stretch", fg='black', font=("Helvetica", 10),anchor="w")
        self.reta_lbl.place(x=25,y=250)
        self.reta=tk.Entry(instance)
        self.reta.place(x=100,y=250)

        self.rxi_lbl=tk.Label(instance, text="Xi stretch", fg='black', font=("Helvetica", 10),anchor="w")
        self.rxi_lbl.place(x=25,y=300)
        self.rxi=tk.Entry(instance)
        self.rxi.place(x=100,y=300)

        self.omega_lbl=tk.Label(instance, text="Omega", fg='black', font=("Helvetica", 10),anchor="w")
        self.omega_lbl.place(x=25,y=350)
        self.omega=tk.Entry(instance)
        self.omega.place(x=100,y=350)

        self.a_lbl=tk.Label(instance, text="Control\nfunction a", fg='black', font=("Helvetica", 10),anchor="w")
        self.a_lbl.place(x=25,y=400)
        self.a=tk.Entry(instance)
        self.a.place(x=100,y=400)

        self.c_lbl=tk.Label(instance, text="Control\nfunction c", fg='black', font=("Helvetica", 10),anchor="w")
        self.c_lbl.place(x=25,y=450)
        self.c=tk.Entry(instance)
        self.c.place(x=100,y=450)

        self.aa_lbl=tk.Label(instance, text="Control\nfunction aa", fg='black', font=("Helvetica", 10),anchor="w")
        self.aa_lbl.place(x=25,y=500)
        self.aa=tk.Entry(instance)
        self.aa.place(x=100,y=500)

        self.cc_lbl=tk.Label(instance, text="Control\nfunction cc", fg='black', font=("Helvetica", 10),anchor="w")
        self.cc_lbl.place(x=25,y=550)
        self.cc=tk.Entry(instance)
        self.cc.place(x=100,y=550)

        self.pl_lbl=tk.Label(instance, text="Poisson?", fg='black', font=("Helvetica", 10),anchor="w")
        self.pl_lbl.place(x=25,y=600)
        self.pl=ttk.Combobox(instance, values=("yes","no"))
        self.pl.place(x=100, y=600)

        self.out=st.ScrolledText(instance, height = 8)
        self.out.place(x=100, y=700)

        self.imgframe=tk.Frame(instance,width=600,height=600,highlightbackground="black",highlightthickness=1)
        self.imgframe.place(x=300,y=100)

        self.display = tk.Label(self.imgframe,image=self.img)
        self.display.pack()
    def comp_grid(self):
        self.out.insert(tk.END,comp_grid(int(self.eta.get()),int(self.xi.get()))+"\n")
        self.img = ImageTk.PhotoImage(Image.open("comp_grid.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()
    def alg_grid(self):
        self.out.insert(tk.END,alg_grid(int(self.eta.get()),int(self.xi.get()),np.double(self.fc.get()),np.double(self.reta.get()),np.double(self.rxi.get()))+"\n")
        self.img = ImageTk.PhotoImage(Image.open("alg_grid.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()
    def elip_grid(self):
        self.out.insert(tk.END,elip_grid(int(self.eta.get()),int(self.xi.get()),np.double(self.fc.get()),np.double(self.reta.get()),np.double(self.rxi.get()),np.double(self.omega.get()),np.double(self.a.get()),np.double(self.c.get()),np.double(self.aa.get()),np.double(self.cc.get()),str(self.pl.get()))+"\n")
        self.img = ImageTk.PhotoImage(Image.open("elip_grid.jpg").resize((575,575),Image.ANTIALIAS))
        self.display.configure(image=self.img,anchor="center")
        self.display.pack()
    
        


window = tk.Tk()
mywin=Window(window)
window.title('TFI')
window.geometry("1000x800+100+100")
window.mainloop()

