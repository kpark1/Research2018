#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from itertools import product
import csv
import random
#-------------------------------------------------------------------------------
#pulse
x = np.array(np.linspace(0, 1000, 1000))
mpv_x = np.loadtxt('mpv_data.csv', unpack = True, delimiter = ',')
mpv = random.choice(mpv_x)
mpv2= mpv/2
eta_x = np.loadtxt('eta_data.csv', unpack = True, delimiter = ',')
eta = random.choice(eta_x)
eta2 = eta/2
A_x = np.loadtxt('A_data.csv', unpack = True, delimiter = ',')
A = random.choice(A_x)
A2= A/2
jitter_x = np.loadtxt('jitter_data.csv', unpack = True, delimiter = ',')
jitter = random.choice(jitter_x)

parameter = [mpv, eta, A]
parameter2= [mpv2, eta2, A2]

y = pylandau.landau(x, *parameter)
y2 = pylandau.landau(x, *parameter2)

new_y = [value + np.random.normal(0, np.sqrt(jitter)) for value in y]
new_ym = [y2[i] + new_y[i] for i in range(len(y))]

#plt.plot(x, new_ym, color = 'b', label = 'data')
#plt.plot(x, y, color ='b')
plt.xlabel('Time [nanoseconds]')
plt.ylabel('Voltage [V]')
plt.title('mpv, eta, A, jitter = '+str(parameter[0])+', '+ str(parameter[1])+', '+ str(parameter[2])+', '+ str(np.around(jitter, decimals = 6)))

#Interpolation
new_x =  np.array(np.linspace(0, 1000, 30))
intp = np.interp(new_x, x, new_ym)
twominindex= argrelextrema(intp, np.less, order  =2)
min_ind = np.ndarray.tolist(twominindex[0])
print('Indices at local minima are ' + str(min_ind))
min_x =[int(new_x[i]) for i in min_ind]
min_y = [int(intp[i]) for i in min_ind]
plt.plot(new_x,intp, 'm', label = 'Interpolation')

#find the first peak parameters
full_intp_list=[]
cut_x_list=[]
cut_y_list = []
par_list = []
for i in range(len(min_x)):
    print('i:'+str(i))
    cut_xx= []
    cut_yy=[]
    if i == 0:
        cut_x=x[0:min_x[i]]
        cut_y= new_ym[0:min_x[i]]
        cut_xx.append(cut_x)
        cut_yy.append(cut_y)
       
    else:
       cut_x= x[min_x[i-1]:min_x[i]]
       cut_y = new_ym[min_x[i-1]:min_x[i]]
       cut_xx.append(cut_x)
       cut_yy.append(cut_y)
    
    cut_x = [int(cut_xx_val) for cut_xx_val in cut_xx[0]]
    cut_y= [float(cut_yy_val) for cut_yy_val in cut_yy[0]]
    n= len(new_ym)-len(cut_y)
    b= [float(cut_y[-1])]*n
    cut_x_list.append(cut_x)
    cut_y_list.append(cut_y)
    full = cut_y + b
    full_intp_list.append(full)
    y= full
    mpv = y.index(max(y))
    
    rmin= 1
    landau_par_rmin, pcov_rmin = curve_fit(pylandau.landau, x, y, p0= (mpv, rmin, rmin))
    array_1 = np.ndarray.tolist(np.around(landau_par_rmin, decimals = 3))
    param_list = [array_1]
    workinglandaupar = [array_1]
    def diff_sq_fn(parameter):
        return round(sum((y-pylandau.landau(x, *parameter))**2), 4)   
    initial_diff_sq = [diff_sq_fn(landau_par_rmin)]
    plt.plot(x,new_ym, 'r')
    plt.plot(cut_x, cut_y,'m')
    plt.plot(x, y, 'g')
    plt.show()
#only one best possible parameter with the least diff_sq
    for eta, A in product(np.linspace(rmin, 25,5 ), np.linspace(rmin, 25, 5)):
        try:
            landau_par, pcov = curve_fit(pylandau.landau, x, y, p0= (mpv, eta,A))
            landau_par = np.ndarray.tolist(np.around(landau_par, decimals =3))
            diff= diff_sq_fn(landau_par)
            par = param_list[0]
                
            if initial_diff_sq[0] < .01:
                par_list.append(landau_par)
                break
            elif landau_par != par and diff < initial_diff_sq[0]:
                param_list.append(landau_par)
                workinglandaupar[0]=landau_par
                initial_diff_sq[0] = diff  
                par_list.append(landau_par)
                break
            else:      
                continue
        except RuntimeError as message:
            print(str(message))
            continue
        except TypeError as message:
            print(str(message))
        except ValueError as message:
            print(str(message))
    #print(par_list, par_list[0])
print(par_list)

plt.plot(cut_x_list[0],cut_y_list[0], 'r',alpha= .2,label= 'interpolationcut')
#plt.plot(cut_x_list[1],cut_y_list[1], '4',alpha= .2, label= 'interpolationcut')
plt.plot(x, full_intp_list[0], 'g',alpha = .2, label = 'interpolation')
#plt.plot(x, full_intp_list[1], 'g',alpha = .2, label = 'interpolation2')
plt.plot(x, pylandau.landau(x, *par_list[0]),'k', alpha = .2,linewidth = 2, label = 'curvefit')
#plt.plot(x, pylandau.landau(x, *par_list[1]), 'y', alpha = .2,linewidth = 2, label = 'curvefit2')
plt.legend()


plt.show()
