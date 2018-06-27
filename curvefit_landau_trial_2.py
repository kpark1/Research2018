#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, all_class, all_class_simulation, Scope2, efficiency
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from itertools import product
import csv
#------------------------------------------------------------------------------

directory = "/home/kpark1/Work/SLab/data_4/"
d=Scope2.ScopeData(directory)
d.save_parameters(plotting = False)


#Plot a histogram of jitter values
d.histogram('jitter')
plt.show()
#jitter -6
#eta -8
#mpv time range
#amplitude .1

Scope2.simulate_pulses(1, np.linspace(0, 2e-06, 2500), [5e-08,0] ,[1e-01,0],[2e-06, 0], plotting = True, plot_pulse = True)
plt.show()
sys.exit()
directory = "/home/kpark1/Work/SLab/data/"
d=Scope2.ScopeData(directory)

for file in sorted(os.listdir(directory)):
    data = d.data_read(file)
    print(file)

#Find baseline and jitter
    t= Scope2.ScopeTrace(data)
    baseline, jitter =t.find_baseline_and_jitter(0,250)
    
#Read CSV file 
directory = "/home/kpark1/Work/SLab/data/"
d=Scope2.ScopeData(directory)

for file in sorted(os.listdir(directory)):
    data = d.data_read(file)
    print(file)

#Find baseline and jitter
    t= Scope2.ScopeTrace(data)
    t.residual()
    plt.show()

sys.exit()
directory = "/home/kpark1/Work/SLab/sim_data_35/"
d=Scope2.ScopeData(directory)

for file in sorted(os.listdir(directory)):
    data = d.data_read(file)
    print(file)
    
#Find baseline and jitter
    efficiency.Efficiency(data).plot()
    plt.xlim(0,1e-7) 
    plt.show()
    efficiency.Efficiency(data).evaluate(data)



#Simulate pulses 
Scope2.simulate_pulses(1, np.linspace(0, 2e-06, 2500), [5e-08,0] ,[1e-01,0],[2e-06, 0], plotting = True, plot_pulse = True)
plt.show()


#Save parameters; In this case, saving data to SLab.
directory = "/home/kpark1/Work/SLab/data_4/"
d=Scope2.ScopeData(directory)
d.save_parameters(plotting = False)


#Plot a histogram of jitter values
#d.histogram('jitter')
#plt.show()
#jitter -6
#eta -8
#mpv time range
#amplitude .1

#Simulate pulses
Scope2.simulate_pulses(2, np.linspace(0, 2e-06, 2500), [5e-08,0] ,[1e-01,0],[2e-06, 0], plotting = True, plot_pulse = True)
plt.show()

#Read CSV file 
directory = "/home/kpark1/Work/SLab/data/"
d=Scope2.ScopeData(directory)

for file in sorted(os.listdir(directory)):
    data = d.data_read(file)
    print(file)

#Find baseline and jitter
    t= Scope2.ScopeTrace(data)
    baseline, jitter =t.find_baseline_and_jitter(0,250)
    parameter = t.parameters()
    print(parameter)
sys.exit()




simfn = all_class.Simulate('jitter_data.csv','mpv_data.csv','eta_data.csv', 'A_data.csv',3)
print(simfn.par_normal())

simfn.histogram()
simfn.simulate_par()
print(range(3))
simfn.simulate_multiplepulses()




directory = "/home/kpark1/Work/SLab/data_4/"
# store parameters of data from a given directory
all_class.Analyze(directory).store_par_j('jfile', 'mfile', 'efile', 'Afile')
sys.exit()

for file in sorted(os.listdir(directory)) :
    ya= all_class.Analyze()
    data = ya.data_read(directory, file)
   
# decode the scope trace
    trace = all_class.ScopeTrace(data)
# find baseline and jitter 
    baseline,jitter = trace.find_baseline_and_jitter()
    #find parameters and plot
    mpv, eta, A = trace.parameters()
    par = [mpv,eta, A]



   
simfn = all_class.Simulate('jitter_data.csv','mpv_data.csv','eta_data.csv', 'A_data.csv',3)
print(simfn.par_normal())

simfn.histogram()
simfn.simulate_par()
print(range(3))
simfn.simulate_multiplepulses()

        
#analyze = all_class_distribution.Analyze('data_4')
#print(analyze.data_read('15192.CSV'))
#analyze.store_par_j('jtrial', 'mtrial', 'etrial', 'Atrial')
analyze = all_class_distribution.Analyze()
for file in sorted(os.listdir("/home/kpark1/Work/SLab/data/")) :
    data = analyze.data_read(file)
# decode the scope trace
    trace = all_class.ScopeTrace(data)
# find baseline and jitter 
    baseline,jitter = trace.find_baseline_and_jitter()
    print(jitter)
    print(trace.parameters())
        

        
data  = Scope2.ScopeFolder('/home/kpark1/SLab/data/', '/home/kpark1/SLab/savedata/data_1_parameters')
print(data.search_pulses([lambda x: x < .002, lambda x: x < .004], ['A', 'mpv'], plotting = False))
