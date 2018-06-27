#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import ScopeTrace
from scipy.optimize import curve_fit
import pylandau

#---------------------------------------------------------------------------------------------------
# M A I N
#---------------------------------------------------------------------------------------------------
# initial settings
mlp.rcParams['axes.linewidth'] = 2

for filename in sys.argv[1:]:
    print ' File to open: ' + filename
    with open(filename,"r") as file:
        data = file.read()
# decode the scope trace
trace = ScopeTrace.ScopeTrace(data,1)

# find baseline and jitter
baseline,jitter = trace.find_baseline_and_jitter(0,250)
#print ' Baseline: %10.6f,  Jitter: %10.6f'%(baseline,jitter)

inverted_yvalues = []
for value in trace.yvalues:
    inverted_yvalue = -(value-baseline)
    inverted_yvalues.append(inverted_yvalue)
#x= np.arange(0,100,0.1)
x = trace.xvalues[0:800] 
x = np.array(x)
y = inverted_yvalues[0:800] 


#initial guess
#mpv, eta, A = 360 ,30, .23: does work
mpv, eta, A = 330, 35, 30
#mpv, eta. A = 300, 50, 40: does not work

coeff, pcov = curve_fit(pylandau.landau, x, y, p0= (mpv, eta, A))
print("Landau parameters are: "+ str(coeff))
y3 = pylandau.landau(x, *coeff)
plt.plot(x, y3,'b',label = 'landau')
plt.plot(x, y, 'k', label = 'data')

mean = sum(inverted_yvalues)/len(trace.xvalues)
sigma = .002
def gaus(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
popt, pcov = curve_fit(gaus, x,y, p0=[1,mean,sigma] )
print('Gaussian parameters are: '+ str(popt))
y2 = gaus(x, *popt)
plt.plot(x, y2, 'r',label = 'Gaussian')

mpv2 = 300
eta2  = 20
A2 = 20
sigma = 10
coeff2, pcov = curve_fit(pylandau.langau, x, y, p0= (mpv2, eta2,sigma, A2))
print("Langau parameters are: "+ str(coeff2))
y4 = pylandau.langau(x, *coeff2)
plt.plot(x, y4,'y',label = 'langau')
plt.legend(loc = 'upper left')
plt.show()

