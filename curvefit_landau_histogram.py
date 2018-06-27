#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, ScopeTrace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from itertools import product
import csv
from collections import Counter

#-------------------------------------------------------------------------------
r_j =  [[.000001, .000004], [.000002, .0000024], [.0000005, .0000035]]
r_m = [[300,350],[304,313], [304, 313]]
r_e =[ [12, 40], [12,40], [12,40]]
r_A = [[0, 0.35], [0,0.35], [0,.35]]

#JITTER
jitter_data_1 = np.loadtxt('jitter_data.csv', unpack = True, delimiter = ',')
jitter_data_1 = np.ndarray.tolist(jitter_data_1)
jitter_data_2 =  np.loadtxt('jitter_data_2.csv', unpack = True, delimiter = ',')
jitter_data_2 = np.ndarray.tolist(jitter_data_2)
jitter_data_3 =  np.loadtxt('jitter_data_3.csv', unpack = True, delimiter = ',')
jitter_data_3 = np.ndarray.tolist(jitter_data_3)
print('<1>\n'+str(sorted(jitter_data_1)))
print('<3>\n'+str(sorted(jitter_data_3)))
def gaus(x, a, x0, sigma):
    return a* np.exp(-.5*((x-x0)/sigma)**2)

def chisq(expt, ob):
    expt = np.ndarray.tolist(np.around(expt))
    ob = np.ndarray.tolist(np.around(ob))
    val_list = []
    for i in range(len(expt)):
        if expt[i]!=0.:
            val = (ob[i]-expt[i])**2/expt[i]
            val_list.append(val)
        else:
            continue
    return sum(val_list)

#data_1

(n,bins, patches)=plt.hist(jitter_data_1, bins = 15, range = r_j[2], color = 'r',edgecolor = 'k', alpha = .5, label = 'Data 1' )
n = np.ndarray.tolist(n)
n.append(0)
n= np.array(n)
a= 12
mean = .0000018
sigma = .00000035
popt, pcov = curve_fit(gaus,bins, n , p0 = [a, mean, sigma])

plt.legend()
dof = len(n)-1
#print('Degrees of freedom for data 1 is ' + str(dof))
#print('The chi squared value for data 1 is ' + str(chisq(gaus(bins,*popt), n)))


#data_3
(n3, bins3, patches3) =plt.hist(jitter_data_3, bins = 15, range = r_j[2] ,color = 'g',edgecolor = 'k',alpha = .5, label ='Data  3')
n3 = np.ndarray.tolist(n3)
n3.append(0)
n3= np.array(n3)
a= 43
mean= .0000023
sigma= .0000002
popt3, pcov2 = curve_fit(gaus,bins3, n3 , p0 = [a, mean, sigma])

#print('Degrees of freedom for data 3 is ' + str(len(n3)-1))
#print('The chi squared value for data 3 is ' + str(chisq(gaus(bins3,*popt3), n3)))
plt.xlim(0, .000005)
plt.ylim(0,60)
plt.xlabel('Jitter Value [V]')
plt.ylabel('Number of Pulses')
plt.legend()
plt.title('Jitter Distribution')
plt.show()

nerror = []
n3error = []
for n_value in n:
    nerror.append(float(np.sqrt(n_value)))

for n3_value in n3:
    n3error.append(float(np.sqrt(n3_value)))
nerror = np.array(nerror)
n3error = np.array(n3error)
#change of units from volt to  microvolt 
bins_after = [bin * (10**6) for bin in bins]
bins3_after = [bin3 * (10**6) for bin3 in bins3]
list1=list(np.linspace(min(bins),max(bins),100))
list1_after = [list1_val *(10**6) for list1_val in list1]
list3=list(np.linspace(min(bins3),max(bins3),100))
list3_after= [list3_val *(10**6) for list3_val in list3]

plt.errorbar(bins_after, n, yerr= nerror, fmt ='o', label = 'Histogram for Data 1')
plt.errorbar(bins3_after, n3, yerr= n3error, fmt = 'o', label = 'Histogram for Data 3')
plt.plot(list1_after,  gaus(list1, *popt), 'k', label = 'Gaussian Fit for Data 1')
plt.plot(list3_after, gaus(list3, *popt3), 'm', label = 'Gaussian Fit for Data 3')
plt.xlabel('Jitter Value [microvolt]')
plt.ylabel('Number of Pulses')
plt.legend()
plt.show()

#PARAMETERS
mpv_data_1 = np.loadtxt('mpv_data.csv', unpack = True, delimiter = ',')
mpv_data_1 = np.ndarray.tolist(mpv_data_1)
mpv_data_2 =  np.loadtxt('mpv_data_2.csv', unpack = True, delimiter = ',')
mpv_data_2 = np.ndarray.tolist(mpv_data_2)
mpv_data_3 =  np.loadtxt('mpv_data_3.csv', unpack = True, delimiter = ',')
mpv_data_3 = np.ndarray.tolist(mpv_data_3)

plt.hist(mpv_data_1, bins = 50, range = [350, 390], color= 'r',edgecolor = 'k',alpha = .5, label = 'Data 1' )
#plt.hist(mpv_data_2, bins = 50, range = r_m[0], color, 'b', edgecolor = 'k',alpha = .5, label = 'Data 2')
plt.hist(mpv_data_3, bins = 50, range = r_m[0], color = 'g',edgecolor = 'k', alpha = .5, label = 'Data 3' )
plt.xlim(300,390)
plt.xlabel('mpv Value')
plt.ylabel('Number of Pulses')
plt.title('mpv Value Distribution')
plt.legend()
plt.show()


eta_data_1 = np.loadtxt('eta_data.csv', unpack = True, delimiter = ',')
eta_data_1 = np.ndarray.tolist(eta_data_1)
eta_data_2 =  np.loadtxt('eta_data_2.csv', unpack = True, delimiter = ',')
eta_data_2 = np.ndarray.tolist(eta_data_2)
eta_data_3 =  np.loadtxt('eta_data_3.csv', unpack = True, delimiter = ',')
eta_data_3 = np.ndarray.tolist(eta_data_3)
plt.hist(eta_data_1, bins = 200, range = r_e[0], color = 'r', edgecolor = 'k',alpha = .5, label = 'Data 1'  )
#plt.hist(eta_data_2, bins = 200, range = r_e[1], color = 'b', edgecolor = 'k', alpha = .5, label = 'Data 2' )
plt.hist(eta_data_3, bins = 200, range = r_e[2], color = 'g', edgecolor = 'k', alpha = .5, label = 'Data 3' )
plt.xlabel('eta Value')
plt.ylabel('Number of Pulses')
plt.title('mpv Value Distribution')
plt.title('eta Value Distribution')
plt.legend()
plt.show()

A_data_1 = np.loadtxt('A_data.csv', unpack = True, delimiter = ',')
A_data_1 = np.ndarray.tolist(A_data_1)
A_data_2 =  np.loadtxt('A_data_2.csv', unpack = True, delimiter = ',')
A_data_2 = np.ndarray.tolist(A_data_2)
A_data_3 =  np.loadtxt('A_data_3.csv', unpack = True, delimiter = ',')
A_data_3 = np.ndarray.tolist(A_data_3)
plt.hist(A_data_1, bins = 200, range = r_A[0], color = 'r', edgecolor = 'k',alpha = .5, label = 'Data 1' )
#plt.hist(A_data_2, bins = 200, range = r_A[1], color = 'b', edgecolor = 'k', alpha = .5, label = 'Data 2' )
plt.hist(A_data_3, bins = 200, range = r_A[2], color = 'g', edgecolor = 'k', alpha = .5, label = 'Data 3'  )
plt.xlabel('A Value')
plt.ylabel('Number of Pulses')
plt.title('A Values Distribution')
plt.legend()
plt.show()
