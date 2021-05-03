import utils_andi
import diffusion_models
import andi
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from pylab import *
import sys
sys.path.insert(0, '../Python')
from functions import *
from flucts import *
from hists import *
from scipy.stats import shapiro
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D 
import csv

def makeplot():
    plt.show()
    plt.close()
    return

def ar1dfa1(s,a,ai=0):
    if(ai!=0):
        a=complex(a,ai)
    f1=(60*s**2*a**2*(a-1)**2-180*s*a**2*(a**2-1)+120*(a**4+a**3+a**2))*(s**4.0-s**2)**(-1.0)
    f0=(s**3*(a-1)**5*(a+1)+15*s**2*a*(a-1)**4)*(s**2-1)**(-1.0)+(-5*s**3*(a-1)**3*(1-7*a-7*a**2+a**3)-15*s**2*a*(a-1)**2*(1-10*a+a**2)+2*s*(a-1)**3*(2-17*a-17*a**2+2*a**3)-120*a**2*(1+a+a**2))*(s**4.0-s**2)**(-1.0)
    F=(a**s*f1+f0)/((-15.0*(a-1)**6))
    return sqrt(F)

def faFL(Y,coverage=0.98):
    l=len(Y)
    N= []



    #create interval lengths to examine
    n=int(l*0.2)
    while(n>2):
        N.append(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l-0.2)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            param=mean(Y[von:bis])
            pv=param
            Fs.append(mean((Y[von:bis]-pv)**2))

        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            param=mean(Y[von:bis])
            pv=param
            Fs.append(mean((Y[von:bis]-pv)**2))
        F.append(sqrt((mean(array(Fs)))))
    return array(N),array(F)

def faFL2d(Y1,Y2,coverage=0.98):
    l=len(Y1)
    N= []

    #create interval lengths to examine
    n=int(l*0.2)
    while(n>2):
        N.append(n)
        #print(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l-0.1)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            Fs.append(mean((Y1[von:bis]-mean(Y1[von:bis]))**2+(Y2[von:bis]-mean(Y2[von:bis]))**2))

        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            Fs.append(mean((Y1[von:bis]-mean(Y1[von:bis]))**2+(Y2[von:bis]-mean(Y2[von:bis]))**2))
        F.append(sqrt((mean(array(Fs)))))
    return array(N),array(F)

def faFL3d(Y1,Y2,Y3,coverage=0.98):
    l=len(Y1)
    N= []

    #create interval lengths to examine
    n=int(l*0.2)
    while(n>2):
        N.append(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l-0.1)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            Fs.append(mean((Y1[von:bis]-mean(Y1[von:bis]))**2+(Y2[von:bis]-mean(Y2[von:bis]))**2+(Y3[von:bis]-mean(Y3[von:bis]))**2))

        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            Fs.append(mean((Y1[von:bis]-mean(Y1[von:bis]))**2+(Y2[von:bis]-mean(Y2[von:bis]))**2+(Y3[von:bis]-mean(Y3[von:bis]))**2))
        F.append(sqrt((mean(array(Fs)))))
    return array(N),array(F) 

def dfaFL(ordnung,ts,q=1,coverage=0.95):
    l=len(ts)
    N= []

    #anomaly
    Y=cumsum(ts)

    #create interval lengths to examine
    n=int(l*0.2)
    while(n>3):
        N.append(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l*0.2)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            param=polyfit(arange(von,bis),Y[von:bis],ordnung, full=True)[0]
            pv=param[-1]
            for j in range(ordnung):
                pv+=param[j]*arange(von,bis)**(ordnung-j)
            Fs.append(mean((Y[von:bis]-pv)**2))
        #print(cumsum(array(Fs)))
        #scaling(cumsum(array(Fs)))
        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            param=polyfit(arange(von,bis),Y[von:bis],ordnung, full=True)[0]
            pv=param[-1]
            for j in range(ordnung):
                pv+=param[j]*arange(von,bis)**(ordnung-j)
            Fs.append(mean((Y[von:bis]-pv)**2))
        F.append(sqrt((mean(array(Fs)))))
    #pylab.legend()
    #pylab.show()
    return array(N),array(F)

def dfa1FL2d(ts1,ts2,coverage=0.95):
    l=len(ts1)
    N= []

    #anomaly
    Y1=cumsum(ts1)
    Y2=cumsum(ts2)

    #create interval lengths to examine
    n=int(l*0.2)
    while(n>3):
        N.append(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l-0.1)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            m1,b1=polyfit(arange(von,bis),Y1[von:bis],1)
            m2,b2=polyfit(arange(von,bis),Y2[von:bis],1)
            pv1=b1+m1*arange(von,bis)
            pv2=b2+m2*arange(von,bis)
            Fs.append(mean((Y1[von:bis]-pv1)**2+(Y2[von:bis]-pv2)**2))

        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            m1,b1=polyfit(arange(von,bis),Y1[von:bis],1)
            m2,b2=polyfit(arange(von,bis),Y2[von:bis],1)
            pv1=b1+m1*arange(von,bis)
            pv2=b2+m2*arange(von,bis)
            Fs.append(mean((Y1[von:bis]-pv1)**2+(Y2[von:bis]-pv2)**2))
        F.append(sqrt((mean(array(Fs)))))
    return array(N),array(F)

def dfa1FL3d(ts1,ts2,ts3,coverage=0.95):
    l=len(ts1)
    N= []

    #anomaly
    Y1=cumsum(ts1)
    Y2=cumsum(ts2)
    Y3=cumsum(ts3)

    #create interval lengths to examine
    n=int(l*0.2)
    while(n>3):
        N.append(n)
        n=int(n*coverage)

    F=[]
    for n in N:
        Fs=[]
        Ns=int((l-0.1)/n)
        #von vorne
        for i in range(Ns):
            von=i*n
            bis=(i+1)*n
            m1,b1=polyfit(arange(von,bis),Y1[von:bis],1)
            m2,b2=polyfit(arange(von,bis),Y2[von:bis],1)
            m3,b3=polyfit(arange(von,bis),Y3[von:bis],1)
            pv1=b1+m1*arange(von,bis)
            pv2=b2+m2*arange(von,bis)
            pv3=b3+m3*arange(von,bis)
            Fs.append(mean((Y1[von:bis]-pv1)**2+(Y2[von:bis]-pv2)**2+(Y3[von:bis]-pv3)**2))

        #von hinten
        for i in range(Ns):
            von=l-(i+1)*n
            bis=l-i*n
            m1,b1=polyfit(arange(von,bis),Y1[von:bis],1)
            m2,b2=polyfit(arange(von,bis),Y2[von:bis],1)
            m3,b3=polyfit(arange(von,bis),Y3[von:bis],1)
            pv1=b1+m1*arange(von,bis)
            pv2=b2+m2*arange(von,bis)
            pv3=b3+m3*arange(von,bis)
            Fs.append(mean((Y1[von:bis]-pv1)**2+(Y2[von:bis]-pv2)**2+(Y3[von:bis]-pv3)**2))
        F.append(sqrt((mean(array(Fs)))))
    return array(N),array(F) 


def PLfit(N,F,abw=0.3):
    abw=1-abw
    #print(abw)
    m=0
    b=0
    textende=''
    textmitte=''
    m,b=polyfit(log10(N)[F>0],log10(F)[F>0],1)
    return m



def FfitBM(N,F,xscale=1,yscale=1,druck=True):
    def AR1fl(x):
        return sum((log(sqrt(x[0])*ar1dfa1(N,0.99)/F))**2)
    x0=[0.000015]
    res=minimize(AR1fl,x0, method='Nelder-Mead')
    #print(res)
    V=res.x[0]
    return V


def FfitWN(N,F):
    def AR1fl(x):
        return sum((log(sqrt(x[0])*ar1dfa1(N,0.0)/F))**2)
    x0=[0.000015]
    res=minimize(AR1fl,x0, method='Nelder-Mead')
    V=res.x[0]
    return V

def scaling(liste, x0=0, delx=1,schrift=''):

    y=array(liste)
    x=x0+arange(1,len(y)+1)*delx

    ly=log10(abs(y)[y>0])
    lx=log10(x[y>0])
    try:
        m,b=pylab.polyfit(lx[:], ly[:], 1)
        return m
    except:
        print("No scaling")
        return 0.75

    

#################################################################################################################
# Creating example trajectories: 
################################################################################################################# 

fale = open("./data/task1.txt", "w")


alles = csv.reader(open("./data/challenge_for_scoring/task1.txt",'r'), delimiter=';', lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
print(alles)

for trajs in enumerate(alles):
    i=trajs[1][0]
    Traj=array(trajs[1][1:])
    if(i==1):
        v=abs(Traj[1:]-Traj[:-1])

        N,F=dfaFL(1,Traj)
       
        N0,F0=faFL(Traj[::10])

        try:
            Fs=zeros(20)
            Ns=20
            n=len(Traj)//20
        
            for j in range(Ns):
                von=j*n
                bis=(j+1)*n
                Fs[j]=mean((Traj[von:bis]-mean(Traj[von:bis]))**2)
            dfl=scaling(cumsum(Fs))/2.0
        except:
            dfl=0.5
    elif(i==2):
        Traj1=(Traj[:len(Traj)//2])
        Traj2=(Traj[len(Traj)//2:])

        v=sqrt((Traj1[1:]-Traj1[:-1])**2+(Traj2[1:]-Traj2[:-1])**2)

        N,F=dfa1FL2d(Traj1,Traj2)
        
        N0,F0=faFL2d(Traj1[::10],Traj2[::10])

        try:
            Fs=zeros(20)
            Ns=20
            n=len(Traj1)//20
        
            for j in range(Ns):
                von=j*n
                bis=(j+1)*n
                Fs[j]=mean((Traj1[von:bis]-mean(Traj1[von:bis]))**2+(Traj2[von:bis]-mean(Traj2[von:bis]))**2)
            dfl=scaling(cumsum(Fs))/2.0
        except:
            dfl=0.5
    else:
        Traj1=(Traj[:len(Traj)//3])
        Traj2=(Traj[len(Traj)//3:2*len(Traj)//3])
        Traj3=(Traj[2*len(Traj)//3:])

        v=sqrt((Traj1[1:]-Traj1[:-1])**2+(Traj2[1:]-Traj2[:-1])**2+(Traj3[1:]-Traj3[:-1])**2)

        N,F=dfa1FL3d(Traj1,Traj2,Traj3)
        
        N0,F0=faFL3d(Traj1[::10],Traj2[::10],Traj3[::10])

        try:
            Fs=zeros(20)
            Ns=20
            n=len(Traj1)//20
        
            for j in range(Ns):
                von=j*n
                bis=(j+1)*n
                Fs[j]=mean((Traj1[von:bis]-mean(Traj1[von:bis]))**2+(Traj2[von:bis]-mean(Traj2[von:bis]))**2+(Traj3[von:bis]-mean(Traj3[von:bis]))**2)
            dfl=scaling(cumsum(Fs))/2.0
        except:
            dfl=0.5
     
    
    N,F=dfaFL(1,Traj)
     
    PVA=FfitBM(N[:3],F[:3])
    WVA=FfitWN(N[-1:],F[-1:])
    PvA=sqrt(PVA*(1-0.99**2))

    if(mean(abs(v))>PvA):
        Nvar=mean(v**2)-PvA
        Nstd=mean(abs(v))-sqrt(PvA)
    else:
        Nvar=0
        Nstd=0

    try:
        J=PLfit(N,sqrt(F**2-0.5*(WVA)*ar1dfa1(N,0.0)**2))-1
    except:
        J=0.5
        
    if(J<0.1):
        J=0.5
    if(J>1):
        J=0.75
    if(isnan(J)):
        J=0.5

    try:
        M=scaling((cumsum(abs(v)-Nstd)))-0.5
    except:
        M=0.5
    try:
        L=scaling((cumsum((v)**2-Nvar)))/2.0
    except:
        L=0.5

    if(sign(M-0.5)!=sign(L-0.5)):
        M=0.5
        L=0.5
    elif(abs(M-0.5)>abs(L-0.5)):
        M=L
    else:
        M=(M+L)/2.0
    
    if(M>1):
        M=0.5
    if(M<0):
        M=0.25

    
    #pylab.legend()
    #pylab.show()
    pylab.close()
 
    try:
        J0=PLfit(N0,F0)
    except:
        J0=0.5
    if(J0<0.1):
        J0=0.5
    if(J0>1):
        J0=0.75
    if(isnan(J0)):
        J0=0.5
    
    if(dfl<0.1):
        dfl=0.25
    if(dfl>1):
        dfl=0.5
    if(isnan(dfl)):
        dfl=0.5
        
    LM=M+dfl
    Jo=J+J0
    
    if(sign(LM-1)==sign(Jo-1)):
        if(abs(LM-1)>abs(Jo-1)):
            Jo=1
        else:
            LM=1
    
    result=Jo+LM-1
    
    if(result>2):
        result=1.5
    if(result<0):
        result=0.5
    
    print(result,int(100*J),int(100*J0),int(100*M),int(100*dfl))
    fale.write(str(i)+'; '+str(result)[:7]+'\n')
    
fale.close()

