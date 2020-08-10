#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np   #perform array like ops
import pandas as pd  #for loading data file like csv, xlsx
from matplotlib import pyplot as plt #plot
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import odeint as Int #solve ode
from scipy.optimize import curve_fit  as fit #to fit
import math


# In[ ]:


## since "Qatar" has relatively small population and cases per 1M is highest we will use this in our network model as
## as it is a nice model with well mixed.
#Total_Population = S0 = 2807805
#Returns required data and the population
def get_data():
    
    Dat,N = pd.read_csv('Data/Kaggle_Covid_19_Qatar_cleaned.csv',sep=','),2807805
    Dat.columns = ['Sl','Date','Infected','Deaths','Recovered']
    Dat['Removed'] = (Dat.loc[:,'Recovered']+Dat.loc[:,'Deaths'])
    Dat['Active_cases'] = (Dat['Infected']-Dat['Removed'])
    
    return N,Dat


# In[ ]:


#Plotting the raw data
def plot_data(Data):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15,5))
    len_data = len(Data)
    days = np.linspace(1,len_data,len_data)
    lst = ['Infected','Active_cases','Removed']
    
    for i in range(3):
        y = np.array(Data.loc[:,lst[i]])
        axes[i].bar(days,y,color = 'y')
        axes[i].set_xlabel("Days")
        axes[i].set_ylabel(lst[i])
        axes[i].grid()
    figure.tight_layout(rect=[0,0,1,0.95])
    


# In[ ]:


# beta controls S to E, sigma= E to I,gamma= I to R.
# defining the system of ODE defining SEIR model    

def system_ODE(y,t, beta, sigma, gamma):
    N=2807805
    s, e, i, r = y
    dydt = [-beta*s*i, beta*s*i-sigma*e, sigma*e-gamma*i, gamma*i]
    return dydt


# In[ ]:


# Case 1: t_train = len(Dat)
# This basically means plot the entire along with ODE plotted for the fit parameter
# Case 2: t_train <<len(Dat)
# Takes fit parameters and plots the ODE along with the entire data

# Y0 : the initial conditions for the ODE
# Data :  to be plotted
# arr_t : Since unit of time for odeint is not days, to fit the data we need to tune this too.It can be normalised but it is yet to be done
# ode_p0 : parameters of the ODE, beta sigma gamma in a tuple
# t_extend : We will plot ode for len(dat)+t_extend number of days

def fit_data(Y0,Data,arr_t,ode_p0,t_extend,t_train):
    
    active = np.concatenate((np.array(Data.loc[0:t_train,'Active_cases']),np.zeros(t_extend)),axis=0)
    removed = np.concatenate((np.array(Data.loc[0:t_train,'Removed']),np.zeros(t_extend)),axis=0)
    total = np.concatenate((np.array(Data.loc[0:t_train,'Infected']),np.zeros(t_extend)),axis=0)
    dead = np.concatenate((np.array(Data.loc[0:t_train,'Deaths']),np.zeros(t_extend)),axis=0)
    recover = np.concatenate((np.array(Data.loc[0:t_train,'Recovered']),np.zeros(t_extend)),axis=0)
    
    n0 = len(active)
    t_dat = np.linspace(0,n0-1,n0)
    t_ode = np.linspace(0,arr_t,n0)
    t_active = np.linspace(0,arr_t+1000,n0)
    #t_active = t_ode
    soln = Int(system_ODE,Y0,t_ode,args=ode_p0)
    soln2 = Int(system_ODE,Y0,t_active,args=ode_p0)
    
    frac_dead = 0
    t=100
    for i in range(len(dat)-t):
        frac_dead += (Data.iloc[i+t,3])/(Data.iloc[i+t,4])
    dead_solve = (frac_dead/(len(dat)-t)) * soln[:,3]
    frac_dead = frac_dead/(len(dat)-t)
    
    fig,a =  plt.subplots(2,2,figsize=(15,15))
    
    a[0][0].bar(t_dat,active,label = 'Observed')
    a[0][0].plot(t_dat,soln2[:,2],'r',label = 'Model')
    a[0][0].legend(loc='best')
    a[0][0].set_xlabel('Days')
    a[0][0].set_ylabel('Number of Active cases/day')
    a[0][0].set_title('Active Cases')

    a[0][1].bar(t_dat,recover,label = 'Observed')
    a[0][1].plot(t_dat,soln[:,3] - dead_solve,'r',label = 'Model')
    a[0][1].legend(loc='best')
    a[0][1].set_xlabel('Days')
    a[0][1].set_ylabel('Number of Recovered cases/day')
    a[0][1].set_title('Recovered')

    a[1][0].bar(t_dat,total,label = 'Observed')
    a[1][0].plot(t_dat,soln[:,2]+soln[:,3],'r',label = 'Model')
    a[1][0].legend(loc='best')
    a[1][0].set_xlabel('Days')
    a[1][0].set_ylabel('Total cases till date')
    a[1][0].set_title('Total Cases')

    a[1][1].bar(t_dat,dead,label = 'Observed')
    a[1][1].plot(t_dat,dead_solve,'r',label = 'Model')
    a[1][1].legend(loc='best')
    a[1][1].set_xlabel('Days')
    a[1][1].set_ylabel('Number of Dead/day')
    a[1][1].set_title('Death')
    
    return frac_dead
    
   


# In[ ]:


# Y0 : the initial conditions for the ODE
# para : parameters of the ODE, beta sigma gamma in a tuple
# t_ld : day of lock down
# arr_t : Since unit of time for odeint is not days, to fit the data we need to tune this too.It can be normalised but it is yet to be done
# n0 : We will plot ode for n0 number of days
# r0 : initial reproduction number
# k : beta_0 = r_0*gamma*k (Refer to report section 3.4.1)
# status_tld : alpha_0 , initial lockdown parameter (Refer to report section 3.4.1)
# frac_dead : fraction of dead cases in removed compartment

# [T,A,R,D] = array with total, active, recovered and dead cases from the ode soln for initial beta.
# ini_ld = SEIR at time of lockdown which serves as initial conditions for solving for variuos alpha after t_ld
# tode_ld = converting unit of time form days to what odeint wants
# ld = list of different lock down parameters
# est = solution of ode for different lockdown parameters
def lock(Y0,para,t_ld,arr_t,n0,r0,k,status_tld,frac_dead):
    
    (beta,sigma,gamma) = para
    t = np.linspace(0,arr_t,n0)
    soln = Int(system_ODE,Y0,t,args=para)
    D = frac_dead*soln[:,3]
    R = soln[:,3]-D
    T = (soln[:,2]+soln[:,3])
    A = (soln[:,2])
    given = [T,A,R,D]
    
    tode_ld = math.floor((arr_t*t_ld)/n0)
    t_preLD = np.linspace(0,tode_ld,t_ld)
    soln_preLD = Int(system_ODE,Y0,t_preLD,args=para)
    
    #index of t_ld is (t_ld -1)
    [ini_ld] = soln_preLD[(t_ld-1):,]
    ld = np.linspace(0,1,11)
    t_postLD = np.linspace(tode_ld,arr_t,n0-t_ld)
    soln_postLD = []
    [total,active,recov,dead]=[[],[],[],[]]
    
    # solving ODE for various lockdown parameters 
    for i in range(len(ld)):
        
        beta = (gamma*(r0/Y0[0])*k)*((1-ld[i])/(1-status_tld))*1
        para_i = (beta,sigma,gamma)
        soln_i = Int(system_ODE,ini_ld,t_postLD,args=para_i)
        soln_postLD.append(soln_i)
        
        d_i = frac_dead*soln_i[:,3]
        r_i = soln_i[:,3]-d_i
        
        total.append(soln_i[:,2]+soln_i[:,3])
        active.append(soln_i[:,2])
        dead.append(d_i)
        recov.append(r_i)
    
    est = [total,active,recov,dead]
    labels = ['Total Cases','Active Cases','Recovered Cases','Dead Cases']
    
    #plotting the solutions
    for i in range(4):
        
        plt.plot(t,given[i],'*',label = 'Current=%s' %(status_tld))
        plt.title(labels[i])
        plt.xlabel('Time')
        plt.ylabel('Number of Cases')
        for j in range(len(ld)):
            array = np.asarray(est[i][j:j+1])
            a_plot = array.reshape(np.shape(t_postLD))
            plt.plot(t_postLD,a_plot,label = '%s' %(round(ld[j],1)))
        plt.legend(loc = 'best')
        plt.show()
       
    


# In[ ]:


## To fit training data before peak.
S0,dat = get_data()
E0 = 0
I0 = dat['Infected'][0]
R0 = 0
r0=4 
ini = [S0,E0,I0,R0]

tode = len(dat)+2200
t_extend = 100+38
n0 = t_extend +len(dat)

K = 0.316
sigma = 1/10
gamma = 1/15
beta = gamma*(r0/S0)*K
param=(beta,sigma,gamma)
status = 0.45


t=100
frac_dead = ((dat.iloc[-1,3])/(dat.iloc[-1,4]))/S0
lock(ini,param,75,tode,n0,r0,K,status,frac_dead) 

train_size = len(dat)  
fit_data(ini,dat,tode,param,t_extend1,train_size)


# In[ ]:


##predicted...

d_frac_prediction = fit_data(ini,dat,tode,param1,t_extend,train_size)


# In[ ]:


# plotting results in different formats
para = (beta,sigma,gamma)
t_ode = np.linspace(0,tode+t_extend,len(dat))
t_days = np.linspace(0,len(dat)+t_extend,len(dat))    
soln = Int(system_ODE,ini,t_ode,args=para)

plt.plot(t_days,soln[:,0],'g',label = 'S(t)')
plt.plot(t_days,soln[:,1],'b',label = 'E(t)')
plt.plot(t_days,soln[:,2],'r',label = 'I(t)')
plt.plot(t_days,soln[:,3],'k',label = 'R(t)')
plt.legend(loc='best')
plt.grid()
plt.show()

plt.plot(t_days,soln[:,2]+soln[:,3],'g',label = 'Total Cases')
plt.plot(t_days,frac_dead*soln[:,3],'b',label = 'Dead')
plt.plot(t_days,(1-frac_dead)*soln[:,3],'r',label = 'Recovered')
plt.xlabel('Days')
plt.ylabel('Number of cases')
plt.legend(loc='best')
plt.grid()

   


# In[ ]:





# In[ ]:





# ## ROUGH

# error = np.sqrt(np.mean(np.square(bar_p[0:140]-soln[:,3][0:140])))
# error
# #bar_p[0:140]
# #soln[:,3][0:140]

# r0 = 1.03
# sigma = 1/5.2
# gamma = 1/9.5
# beta = gamma*(r0/S0)
# params = (0.0000015,sigma,gamma)
# soln1 = Int(system_ODE,y0,t,args=params)
# plt.figure(figsize=(40,10))
# plt.subplot(1,2,2)
# plt.plot(t,soln1[:,0],'b',label = 'S(t)')
# plt.plot(t,soln1[:,1],'r',label = 'E(t)')
# plt.plot(t,soln1[:,2],'g',label = 'I(t)')
# plt.plot(t,soln1[:,3],'y',label = 'R(t)')
# plt.bar(t,bar_p)
# #plt.ylim(0,300000)
# plt.legend(loc='best')
# plt.grid()
# plt.show()

# np.array(Data_new.loc[:,'Infected']).shape
# len_data
# lst = ['Infected','Removed','Active_cases']
# lst[0]
