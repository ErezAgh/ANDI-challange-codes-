import sys
sys.path.insert(0, 'C:/Users/user/Documents/Competition')
import os
import utils_andi
import diffusion_models
import andi
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
import pylab
from pylab import *
from scipy.stats import shapiro
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import minimize
import csv

import math  
import pandas as pd
import glob

sys.path.insert(0, '../Python')
import scipy
from scipy import fftpack  
from scipy.optimize import minimize 


# From the file generating the training set: 
import numpy as np
from pylab import *
import math  
import pandas as pd
import glob

from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib qt5  
import sys 
import utils_andi
import diffusion_models
import andi
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../Python')
from scipy.stats import shapiro
import scipy
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D 
from scipy import fftpack 
from numpy import *
from pylab import * 
from scipy.optimize import minimize
from random import * 
#%% Questions: 

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
        m,b=polyfit(lx[:], ly[:], 1)
        return m
    except:
        print("No scaling")
        return 0.75


def FindExponents(Traj,dim=1) :
    if(dim==1):
        v=Traj[1:]-Traj[:-1]

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
    elif(dim==2):
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

    LM=(L+dfl)/2+0.5
    L=LM-M
    J=(J+J0)/2
    return M,L,J,LM+J-1.0

#%% 
# ## Q1:
# def Q1Erez(M,L,J) : # Opposite of Q2. (> Screens for FBM and LW, < Screens for SBM).
#     try: 
#         if (abs(J/L-1)-abs(M/L-1))>0.15: 
#             return 1 
#         elif (abs(J/L-1)-abs(M/L-1))<-0.15: 
#             return 0 
#         else: 
#             return 2
#     except:
#         print("!!! Error in Q1Erez !!!")
#         return 0
    
##############################
## Q2:
# def Q4Erez(Trajectory,ThresholdFoFatTailness) : # Checks for ``Fat-Tailness".ThresholdForFarTailness=0.15. 
#        increments=Trajectory[1:]-Trajectory[:-1] 
#        FatTailness=np.median(abs(increments))/((max(abs(increments))+min(abs(increments)))/2) # For e.g., FBM, this should bbe large, for CTRW, should be small. 
#        if FatTailness<ThresholdFoFatTailness: 
#            return 1 
#        elif FatTailness>0.175: 
#            return 0 
#        else: 
#            return 2 

def Q4Erez(Trajectory,ThresholdFoFatTailness) : # Checks for ``Fat-Tailness".ThresholdForFarTailness=0.1. 
       increments=Trajectory[1:]-Trajectory[:-1] 
       FatTailness=np.median(abs(increments))/((max(abs(increments))+min(abs(increments)))/2) # For e.g., FBM, this should bbe large, for CTRW, should be small. 
       if FatTailness<ThresholdFoFatTailness or math.isnan(FatTailness): 
           return 1 
       elif FatTailness>0.175: 
           return 0 
       else: 
           return 2 
##############################

## Q3: 
def PlateauDetectionFlatness(Trajectory,ThresholdForLargeJump,ThresholdForCTRW) : # ThresholdForLargeJump=3, ThresholdForCTRW=0.5. 
    try: 
        increments=Trajectory[1:]-Trajectory[:-1] 
        AbsIncrements=np.array(abs(increments))
        XAxis=np.arange(1,len(AbsIncrements)+1); 
        Threshold=ThresholdForLargeJump # Threshold for large jump. 
        MeanAbsincrements=sqrt(np.mean(increments**2)-(np.mean(increments))**2) # np.mean(AbsIncrements)
        XAxisLarge=XAxis[AbsIncrements>Threshold*MeanAbsincrements] 
        if len(XAxisLarge)==0 : 
            return 2 # Could not detect large jumps. - Setting to one for CTRW, will give false ones for FBM now. 
        else: 
            LargeAbsIncrementsDiff=np.diff(XAxis[AbsIncrements>Threshold*MeanAbsincrements]) 
            LargeAbsIncrementsDiff=np.insert(LargeAbsIncrementsDiff,0,XAxisLarge[0])
            ii=0  
            IsPlateauArr=[] 
            qq=1 
            while qq<len(LargeAbsIncrementsDiff)+1 : 
                if (LargeAbsIncrementsDiff[qq-1]!=1) and (len(arange(ii,ii+LargeAbsIncrementsDiff[qq-1]-1))!=1): 
                    m,b=polyfit(arange(ii,ii+LargeAbsIncrementsDiff[qq-1]-1),Trajectory[ii:ii+LargeAbsIncrementsDiff[qq-1]-1],1) 
                    if abs(m)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : 
                        if len(arange(ii,ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)))>1: 
                             m1,b1=polyfit(arange(ii,ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)),Trajectory[ii:ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)],1) 
                             ShortTrajectory=Trajectory[ii:ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)] 
                             if abs(m1)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : # Checking at half-length of the plateau. 
                                 if len(arange(ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)+1,ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1))))>1: 
                                    m2,b2=polyfit(arange(ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)+1,ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1))),Trajectory[ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)+1:ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1))],1) 
                                    ShortTrajectory=Trajectory[ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1)/2)+1:ii+math.floor((LargeAbsIncrementsDiff[qq-1]-1))] 
                                    if abs(m2)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : # Checking at second half-length of the plateau. 
                                        IsPlateauArr.append(1) 
                                    else: 
                                        IsPlateauArr.append(0) 
                                 else: 
                                    IsPlateauArr.append(0) 
                             else: 
                                IsPlateauArr.append(0) 
                        else: 
                            IsPlateauArr.append(0)
                    else: 
                        IsPlateauArr.append(0) 
                else: # If the interval has size 1: 
                    IsPlateauArr.append(0) 
                ii=ii+LargeAbsIncrementsDiff[qq-1] 
                qq=qq+1 
            m,b=polyfit(arange(ii,len(Trajectory)),Trajectory[ii:len(Trajectory)],1) # Last interval in the sequence. 
            if abs(m)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : 
                if len(arange(ii,math.floor((ii+len(Trajectory))/2)))>1: 
                    ShortTrajectory=Trajectory[ii:math.floor((ii+len(Trajectory))/2)] 
                    m1,b1=polyfit(arange(ii,math.floor((ii+len(Trajectory))/2)),Trajectory[ii:math.floor((ii+len(Trajectory))/2)],1) # Half of last interval in the sequence. 
                    if abs(m1)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : 
                         ShortTrajectory=Trajectory[ii:math.floor((ii+len(Trajectory))/2)] 
                         if len(arange(math.floor((ii+len(Trajectory))/2),len(Trajectory)))>1:  
                             m2,b2=polyfit(arange(math.floor((ii+len(Trajectory))/2),len(Trajectory)),Trajectory[math.floor((ii+len(Trajectory))/2):len(Trajectory)],1) # Second half of last interval in the sequence. 
                             if abs(m2)<(abs((Trajectory[0])-(Trajectory[len(Trajectory)-1]))/len(Trajectory)) : 
                                 IsPlateauArr.append(1) 
                             else: 
                                 IsPlateauArr.append(0) 
                         else: 
                            IsPlateauArr.append(0) 
                    else: 
                        IsPlateauArr.append(0) 
                else: 
                    IsPlateauArr.append(0) 
            else: 
                IsPlateauArr.append(0) 
            return(sum(IsPlateauArr[0:len(IsPlateauArr)-1]*LargeAbsIncrementsDiff)+IsPlateauArr[len(IsPlateauArr)-1]*(len(Trajectory)-sum(LargeAbsIncrementsDiff))>ThresholdForCTRW*len(Trajectory))
    except: 
        return 0
       
## Q4:
def hhatCor(J,M,L):
    try:
        if ( (abs(L-0.5)-abs(J-0.5) )<0.07 and (0.5*abs(M-0.5)-abs(J-0.5))<-0.07 ): 
            return 1 
        elif ( (abs(L-0.5)-abs(J-0.5) )>0.07 or (0.5*abs(M-0.5)-abs(J-0.5))>0.07 ): # Since np.quantile(ResultsForStatistics, 0.9)=0.088044 (for J Vs L). Based on 500 paths of length 550, all the H's
            return 0 
        else: 
            return 2
    except:
        print("!!! Error in hatCor !!!")
        return 2


## Q5:
def supCor(J): # 1 if superdiffusive with Joseph, 0 otherwise. 
    try: 
        if (J>0.6): 
            return 1 
        elif J<0.4: 
            return 0 
        else: 
            return 2  
    except:
        print("!!! Error in supCor !!!")
        return 2 


## Q6:
def isInfden(Trajectory):
    T=zeros(len(Trajectory)//10)
    for j in range(len(T)):
        T[j]=mean(Trajectory[j*10:(j+1)*10])
    w=T[1:]-T[:-1]
    try:
        av1=cumsum(abs(w))
        v21=cumsum(w**2)
        av=av1[(av1>0) & (v21>0)]
        v2=v21[(av1>0) & (v21>0)]
        ide=0.4*sqrt(av[len(av)//2:]/sqrt(arange(len(av))[len(av)//2:]/(len(av)))/sqrt(v2[len(av)//2:]))
        sca=av[len(av)//2:]/v2[len(av)//2:]
        # print(var(ide),var(sca),var(ide)>var(sca))
        if (var(ide)<var(sca)): 
            return 0 
        elif (2*var(ide)>var(sca)): 
            return 1 
        else: 
            return 2
    except:
        print("!!! Error in isInfden !!!")
        return 2

## Q7:
def absCor(v): # Detects Levy walk. 
    try:
        N,F1=dfaFL(1,cumsum(v))
        N,F2=dfaFL(1,cumsum(abs(v)-mean(abs(v))))
        F1=F1[0]
        F2=F2[0] 
        if (F1>3.5*F2):
            return 1  
        elif (F1<2.0*F2): 
            return 0 
        else: 
            return 2 
    except:
        print("!!! Error in absCor !!!")
        return 2 
    
## Q8: 
#Plateau detection + sum comparisson:
def PlateauDetectionVarianceComparisson(Trajectory,ThresholdForLargeJump) : # ThresholdForLargeJump=3, ThresholdForCRTWSum=0.1 
    try: 
        ThresholdForCRTWSum=0.1 
        increments=Trajectory[1:]-Trajectory[:-1]
        AbsIncrements=np.array(abs(increments))
        XAxis=np.arange(1,len(AbsIncrements)+1);
        MeanAbsincrements=sqrt(np.mean(increments**2)-(np.mean(increments))**2) # np.mean(AbsIncrements)
        XAxisLarge=XAxis[AbsIncrements>ThresholdForLargeJump*MeanAbsincrements]
        MinimalSegment=int(len(Trajectory)/20)
        if len(XAxisLarge)==0 :
            return (2) # Could not detect large jumps. - Setting to one since for CTRW, will create fales ones for FBM now.
        else:
            LargeAbsIncrementsDiff=np.diff(XAxis[AbsIncrements>ThresholdForLargeJump*MeanAbsincrements]) # Lengths of increments between large jumps.
            LargeAbsIncrementsDiff=np.insert(LargeAbsIncrementsDiff,0,XAxisLarge[0]) # Length of first increment, before first large jumps.
            ii=0  
            IncrementSumArr=[]
            qq=1
            while qq<len(LargeAbsIncrementsDiff)+1 :
        #        if (LargeAbsIncrementsDiff[qq-1]!=1) and (len(arange(ii,ii+LargeAbsIncrementsDiff[qq-1]-1))!=1):
                if (LargeAbsIncrementsDiff[qq-1]>=MinimalSegment) and (len(arange(ii+1,ii+LargeAbsIncrementsDiff[qq-1]-2))!=1):
                    IncrementSum=sum((Trajectory[ii+1:ii+LargeAbsIncrementsDiff[qq-1]-2]))
                    IncrementSumArr.append(IncrementSum)
                ii=ii+LargeAbsIncrementsDiff[qq-1]
                qq=qq+1
            IncrementSum=sum((Trajectory[ii+1:len(Trajectory)]))  # Last interval in the sequence.
            IncrementSumArr.append(IncrementSum)
            Hist=hist(IncrementSumArr)
            if (sum(Hist[0])>2): 
                return (max(Hist[0])>=ThresholdForCRTWSum*sum(Hist[0]))
            elif (sum(Hist[0])==2) and ((abs((Hist[0][0]-Hist[0][1])/Hist[0][0])<=0.1) or (abs((Hist[0][0]-Hist[0][1])/Hist[0][1])<=0.1)): 
                return 1
            else: 
                return 0  
    except: 
        return 2 
    
# Q9: 
def IsSmall(M,L,J) : 
    if 2.5*abs(L-0.5)+abs(J-0.5)<abs(M-0.5): 
        return 1 
    elif (2.5*abs(L-0.5)+abs(J-0.5)<abs(M-0.5))*0.5: 
        return 2 
    else: 
        return 0 
#%% 

def getresponse(Trajectory,Dim): 
    increments=Trajectory[1:]-Trajectory[:-1]
    M,L,J,H=FindExponents(Trajectory)
    # Q1=int(Q1Erez(M,L,J)) 
    Q2=int(Q4Erez(Trajectory,0.125)) 
    Q3=int(PlateauDetectionFlatness(Trajectory,3.0,0.75)) 
    Q4=int(hhatCor(J,M,L)) 
    Q5=int(supCor(J)) 
    Q6=int(isInfden(Trajectory)) 
    Q7=int(absCor(increments)) 
    Q8=int(PlateauDetectionVarianceComparisson(Trajectory,3.0)) 
    Q9=int(IsSmall(M,L,J))
    # return Q1,Q2,Q3,Q8,Q4,Q5,Q6,Q7,Q9 
    return Q2,Q3,Q8,Q4,Q5,Q6,Q7,Q9
#%%
#alles=loadtxt('/home/erezagh/Desktop/Data-Directory/AndiChallange/TrainingAnswersOct30.txt', delimiter=',')
##alles=csv.reader(open("/home/erezagh/Desktop/Data-Directory/AndiChallange/TrainingAnswersOct30.txt",'r'), delimiter=',', lineterminator='\n')
#f = open("/home/erezagh/Desktop/Data-Directory/AndiChallange/task2AnswersTemp.txt", "a")
#
#
#
#daten = csv.reader(open("/home/erezagh/Desktop/Data-Directory/AndiChallange/development_for_scoring_new/task2.txt",'r'), delimiter=';', lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
alles=loadtxt('/home/user/Downloads/TrainingSetOct30/AnswersOct30.txt', delimiter=',')
#alles=csv.reader(open("/home/erezagh/Desktop/Data-Directory/AndiChallange/TrialAnswersLong.txt",'r'), delimiter=',', lineterminator='\n')
f = open("/home/user/Downloads/RealResultsANDIChallange/Oct30/task21DFinal.txt", "a")
# f1 = open("/home/user/Downloads/RealResultsANDIChallange/Oct30/TempToBeDeletedAnswers.txt", "a")



daten = csv.reader(open("/home/user/Downloads/NewOct30/task2.txt",'r'), delimiter=';', lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
print(daten)
ii=1
for trajs in enumerate(daten):
    i=trajs[1][0] 
    ii=ii+1 
    Dim=int(i)
    if (i!=1.0): 
        continue 
    else: 
        print(ii)
        Trajectory=np.array(trajs[1][1:]) 
        answers=getresponse(Trajectory,Dim) 
        # increments=Trajectory[1:]-Trajectory[:-1]
        # M,L,J,H=FindExponents(Trajectory) 
        # Q2=int(Q4Erez(Trajectory,0.125)) 
        # Q3=int(PlateauDetectionFlatness(Trajectory,3.0,0.75)) 
        # Q4=int(hhatCor(J,M,L)) 
        # Q5=int(supCor(J)) 
        # Q6=int(isInfden(Trajectory)) 
        # Q7=int(absCor(increments)) 
        # Q8=int(PlateauDetectionVarianceComparisson(Trajectory,3.0)) 
        # Q9=int(IsSmall(M,L,J)) 
        # QuestionsList=[Q2,Q3,Q8,Q4,Q5,Q6,Q7,Q9]
        # f1.write(str(i)+','+ str(QuestionsList[0])+','+ str(QuestionsList[1])+','+ str(QuestionsList[2])+','+ str(QuestionsList[3])+','+ str(QuestionsList[4])+','+ str(QuestionsList[5])+','+ str(QuestionsList[6])+','+ str(QuestionsList[7])+'\n')

        models=alles[:,0]
        thesemodels=models[(alles[:,1]==answers[0]) & (alles[:,2]==answers[1]) & (alles[:,3]==answers[2]) & (alles[:,4]==answers[3]) & (alles[:,5]==answers[4]) & (alles[:,6]==answers[5]) & (alles[:,7]==answers[6]) & (alles[:,8]==answers[7])]
#    print(thesemodels)
        alres=len(thesemodels)*1.0
        if(alres>0):
            prob0=sum(thesemodels==0)/alres
            prob1=sum(thesemodels==1)/alres
            prob2=sum(thesemodels==2)/alres
            prob3=sum(thesemodels==3)/alres
            prob4=sum(thesemodels==4)/alres
        else: 
            # thesemodels=models[(alles[:,1]==answers[0]) & (alles[:,2]==answers[1]) & (alles[:,3]==answers[2]) & (alles[:,4]==answers[3]) & (alles[:,5]==answers[4]) & (alles[:,6]==answers[5]) & (alles[:,8]==answers[7])]
            thesemodels=models[(alles[:,2]==answers[1]) & (alles[:,3]==answers[2]) & (alles[:,4]==answers[3]) & (alles[:,5]==answers[4]) & (alles[:,6]==answers[5]) & (alles[:,7]==answers[6]) & (alles[:,8]==answers[7])]
            alres=len(thesemodels)*1.0
            if(alres>0):
                prob0=sum(thesemodels==0)/alres
                prob1=sum(thesemodels==1)/alres
                prob2=sum(thesemodels==2)/alres
                prob3=sum(thesemodels==3)/alres
                prob4=sum(thesemodels==4)/alres
            else: 
                thesemodels=models[(alles[:,2]==answers[1]) & (alles[:,3]==answers[2]) & (alles[:,5]==answers[4]) & (alles[:,6]==answers[5]) & (alles[:,8]==answers[7])]
                alres=len(thesemodels)*1.0
                if(alres>0):
                    prob0=sum(thesemodels==0)/alres
                    prob1=sum(thesemodels==1)/alres
                    prob2=sum(thesemodels==2)/alres
                    prob3=sum(thesemodels==3)/alres
                    prob4=sum(thesemodels==4)/alres 
                else: 
                    prob0=0.2 
                    prob1=0.3
                    prob2=0.1
                    prob3=0.2
                    prob4=0.2 
            
        f.write(str(i)+'; '+str(prob0)+'; '+str(prob1)+'; '+str(prob2)+'; '+str(prob3)+'; '+str(prob4)+'\n')

#pred_to_txt = np.ones((len(predictions), 6))
#for i in range(len(predictions)):
#    for j in range(5):
#       
#        pred_to_txt[i, j+1] = predictions[i][j]
#
#np.savetxt('task2.txt', pred_to_txt.astype(float), fmt = '%1.5f', delimiter = ';')

f.close()