#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import ScopeTrace
from scipy.optimize import curve_fit
from scipy import integrate

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
gaussian_values = []
landau_values = []

for value in trace.yvalues:
    inverted_yvalue = -(value-baseline)
    inverted_yvalues.append(inverted_yvalue)

amp = max(inverted_yvalues)
amp_idx= inverted_yvalues.index(amp)

#half maximum
left_inverted = inverted_yvalues[0:amp_idx]
right_inverted = inverted_yvalues[amp_idx:]
right_idx = min(range(len(left_inverted)), key=lambda i:abs(right_inverted[i]-.5*amp))
left_idx   = min(range(len(left_inverted)), key=lambda i:abs(left_inverted[i]-.5*amp))
HWHM =np.absolute(trace.xvalues[left_idx]-trace.xvalues[amp_idx])
width = HWHM/(np.sqrt(2*np.log(2)))
mean = amp_idx

for value in trace.xvalues:    
    gaussian_value =amp* np.exp(-.5*((value-mean)/width)**2)
    gaussian_values.append(gaussian_value)
   # landau_value = amp*np.exp(-.5*(value+np.exp(-value)))
    #landau_value = np.pi**-1* integrate.quad(lambda t: np.exp(-t)*np.cos(t*(value-mean)/amp+2*t*np.log10(t/mean) /np.pi), 0, np.inf)
    #landau_values.append(landau_value)
new_xvalues = list(range(len(trace.yvalues)))

#FUNCTION
def Gaussian(x_par,width_par, mu_par):
    "makes a gaussian function"
    return np.exp(-.5*((x_par-mu_par)/width_par)**2)/(width_par*np.sqrt(2*np.pi))

fitparams, fitcovariance = curve_fit(Gaussian, trace.xvalues, inverted_yvalues)
plt.figure(1)
plt.scatter(trace.xvalues, Gaussian(trace.xvalues,fitparams[0], fitparams[1]))
plt.xlim([-10,10])
plt.ylim([0, 1e-08])

# define the figure
fig = mlp.pyplot.gcf()
fig.set_size_inches(18.5, 14.5)

#FIGURE 1
plt.figure(2)
#plt.plot(trace.xvalues, inverted_yvalues, color = 'r')
#plt.plot(new_xvalues, gaussian_values, color = 'g') 
#plt.plot(new_xvalues, fitted_gaussian_values, color = 'b')
plt.title("Oscilloscope: gaussian_values", fontsize=28)
plt.xlabel('x-interval[n%s] '%trace.horizontal_units, fontsize=24)
plt.ylabel('y-reading [%s]'%trace.vertical_units, fontsize=24)
plt.xlim([200,700])
plt.ylim([-.3,.5])

#FIGURE 2
#plt.figure(3)
#plt.plot(trace.xvalues, inverted_yvalues, color = 'r')
#plt.plot(new_xvalues, landau_values, color = 'g')
#plt.title("Oscilloscope: landau_values", fontsize=28)
#plt.xlabel('x-interval[n%s] '%trace.horizontal_units, fontsize=24)
#plt.ylabel('y-reading [%s]'%trace.vertical_units, fontsize=24)

plt.show()
