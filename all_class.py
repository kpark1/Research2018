#!/usr/bin/env python
import sys, os, pylandau
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import product
from random import choice, uniform
import csv

#-------------------------------------------------------------------------------
class ScopeTrace():
    def __init__(self, data, n_average = 1, xmin=0, directory ="/home/kpark1/Work/SLab/data_4/" ):
        self.data = data
        self.n_average = n_average
        self.directory = directory
        self.xvalues = []
        self.yvalues = []
        self.xmin = xmin
        try: 
            self.sample_interval = self.find_value('Sample Interval', data)
        except: 
            self.sample_interval = 1

        try: 
            self.trigger_point = self.find_value('Trigger Point', data)* self.sample_interval + self.xvalues[0]
        except:
            try: 
                self.trigger_point= self.find_value('Trigger Offset', data)
            except:
                self.trigger_point = 250
        x=0
        y=0
        n=0
        i = 0
        for line in data.split('\n'):
            f = line.split(',')
            if len(f) <5:
                continue
            x+= float(i)
            y+= float(f[4])
            n+=1
            if n>=n_average:
                self.xvalues.append(x/n)
                self.yvalues.append(y/n)
                n= 0
                x= 0
                y= 0
            i +=1

    def trigger_point(self):
        return self.trigger_point   

    def find_baseline_and_jitter(self):
        #to call
        #baseline, jitter = trace.find_baseline_and_jitter
        n = 0
        sum = 0
        sum2 = 0
        
        for x,y in zip(self.xvalues,self.yvalues):

            if x> self.xmin and x< self.trigger_point:
                sum = sum + y
                sum2 = sum2 + y*y
                n = n + 1
        baseline = sum/n
        jitter = sum2/n - baseline*baseline
        return (baseline,jitter)

    def inverted(self):
        baseline,jitter = self.find_baseline_and_jitter()
        return [-(val-baseline) for val in self.yvalues]

    def diff_sq_fn(self, parameter):
        y= np.asarray(self.inverted())
        self.xvalues = np.asarray(self.xvalues)
        return round(sum((y-pylandau.landau(self.xvalues, *parameter))**2), 4)

    def parameters(self):
        #to call
        # trace = all_class.cp.ScopeTrace(data) ...print(trace.parameters())
        #use index number instead of self.xvalues as rescaling enables the pylandau.landau function to find parameters
        x= np.linspace(0, len(self.xvalues)-1, len(self.xvalues))
        y= self.inverted()
        
        #find the x value of the largest peak
        y_array = np.array(y)
        idx = np.where(y_array==y_array.max())
        idx = idx[0][0]
        x_values_peak = x[idx]
        
        #make initial guess for parameters mpv, eta, A, which correspond to the x value, width, and amplitude of the peak
        mpv = x_values_peak
        rmin=1
        landau_par_rmin, pcov_rmin = curve_fit(pylandau.landau, x, y, p0= (mpv, rmin, rmin))
        array_1 = np.ndarray.tolist(np.around(landau_par_rmin, decimals = 3))
        param_list = [array_1]
        workinglandaupar = [array_1]
        initial_diff_sq = [self.diff_sq_fn(landau_par_rmin)]
        
        #checks if initial guess is a good fit and if not, loop through a range of possible parameters to make new guesses until a set of parameters has a decent fit
        for eta, A in product(np.linspace(rmin,25,5), np.linspace(rmin,25,5)):
            try:
                landau_par, pcov = curve_fit(pylandau.landau, x, y, p0= (mpv, eta,A))
                landau_par = np.ndarray.tolist(np.around(landau_par, decimals =3))
                diff= self.diff_sq_fn(landau_par)
                par = param_list[0]
                if initial_diff_sq[0] < .01:
                    break
                elif landau_par != par and diff < initial_diff_sq[0]:
                    param_list.append(landau_par)
                    workinglandaupar[0] = landau_par
                    initial_diff_sq[0] = diff
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
        mpv = workinglandaupar[0][0]*self.sample_interval
        eta = workinglandaupar[0][1]*self.sample_interval
        A = workinglandaupar[0][2]*self.sample_interval
        return (mpv, eta, A) 
    
    
    def plot(self, parameters= None):
        #parameters in an array   
        x= np.array(self.xvalues)
        y = np.array(self.inverted())
        plt.plot(x,y, label = 'Data')
        if (parameters != None) and (len(parameters) ==3):
            plt.plot(x, pylandau.landau(np.linspace(0, len(x)-1, len(x)), parameters[0]/self.sample_interval, parameters[1]/self.sample_interval, parameters[2]/self.sample_interval), label = 'Landau Fit')
            plt.show()

class Analyze():
    def __init__(self, directory="/home/kpark1/Work/SLab/data/"):
        self.directory = directory
         
    def data_read(self, filename):
#filename needs to be in quotation marks (string)
        with open(self.directory+'/'+filename, "r") as file1: 
            data=file1.read()
            print(str(file1) + ':\n')
            return data
    
#after reading files in a directory, store parameters and jitter 
    def store_par_j(self, jitterfilename, mpvfilename, etafilename, Afilename):
#filenames need to be in quotation marks (string)
        jitter_stored = []
        mpv_stored = []
        eta_stored = []
        A_stored = []
        for file in sorted(os.listdir("/home/kpark1/Work/SLab/data_4/")):
            data = self.data_read(file)
#
            par = ScopeTrace(data)
            baseline, jitter = par.find_baseline_and_jitter()
            mpv,eta, A = par.parameters()
            jitter_stored.append(jitter)
            mpv_stored.append(mpv)
            eta_stored.append(eta)
            A_stored.append(A)
 
        with open(jitterfilename, 'wb') as f:
            wr = csv.writer(f)
            wr.writerow(jitter_stored)

        with open(mpvfilename, 'wb') as f:
            wr = csv.writer(f)
            wr.writerow(mpv_stored)

        with open(etafilename, 'wb') as f:
            wr = csv.writer(f)
            wr.writerow(eta_stored)

        with open(Afilename, 'wb') as f:
            wr = csv.writer(f)
            wr.writerow(A_stored)
    
        return (jitterfilename, mpvfilename, etafilename, Afilename)

    #def histogram(self, jitterfile, mpvfile, etafile, Afile):
        
    #def par_read(self, jitterfile, mpvfile, etafile, Afile)
        
    


class Simulate():
    def __init__(self, jitterfile, mpvfile, etafile, Afile, n_pulses=1):
   #ex) mpv list1 = mpv_data.csv
        self.jitterfile = jitterfile
        self.mpvfile = mpvfile
        self.etafile = etafile
        self.Afile = Afile
        self.n_pulses = 1
    
    def gaus(self, x, a, x0, sigma):
        return a* np.exp(-.5*((x-x0)/sigma)**2)

   #read an array of parameters from a list of stored parameters
    def par_read(self):
        jitter_s = np.loadtxt(self.jitterfile, unpack = True, delimiter = ',')
        mpv_s = np.loadtxt(self.mpvfile, unpack = True, delimiter = ',')
        eta_s = np.loadtxt(self.etafile, unpack = True, delimiter = ',')
        A_s = np.loadtxt(self.Afile, unpack = True, delimiter = ',')
        return (jitter_s, mpv_s, eta_s, A_s)

    #randomly pick parameters from normal distribution
    def par_normal(self, mpvmaxlim =400, Amaxlim= 1):
        jitter_s, mpv_s, eta_s, A_s = self.par_read()
        
        jitter = choice(jitter_s)
        while True:
            mpv = choice(mpv_s)
            if mpv < mpvmaxlim:
                break
        eta= choice(eta_s)
        while True:
            A = choice(A_s)
            if A < Amaxlim:
                break
        return (float(jitter), float(mpv), float(eta), float(A))

    #randomly pick parameters from uniform distribution
    def par_uniform(self, mpvmaxlim =400, Amaxlim= 1):
        jitter_s, mpv_s, eta_s, A_s = self.par_read()
        
        jitter = uniform(min(jitter_s), max(jitter_s))
        while True:
            mpv = uniform(min(mpv_s), max(jitter_s))
            if mpv < mpvmaxlim:
                break
        eta= uniform(min(eta_s), max(eta_s))
        while True:
            A = uniform(min(A_s), max(A_s))
            if A < Amaxlim:
                break
        return (float(jitter), float(mpv), float(eta), float(A))
   
   #directoryname is the name of a folder the stored parameters originated from
    def histogram(self, directoryname= 'data', unit = 10**6, a_initial = 12, mean_initial =.0000018, 
                  sigma_initial =.00000035, hbins=15, hrange =[.0000005, .0000035], hcolor= 'r', hedgecolor = 'k', halpha = .5):
        jitter_s, mpv_s, eta_s, A_s = self.par_read()
        jitter= np.ndarray.tolist(jitter_s)
        (n,bins, patches)= plt.hist(jitter, bins = int(hbins), range = hrange, color = hcolor,edgecolor = hedgecolor, alpha = float(halpha))
        
        #len(n)+1 = len(bins) 
        n = np.ndarray.tolist(n)
        n.append(0)
        n= np.array(n)
        nerror = []
        n3error = []
        for nval in n:
            nerror.append(float(np.sqrt(nval)))
        
        nerror = np.array(nerror)

        #change of units from volt to  microvolt 
        bins_conv = [bin * int(unit) for bin in bins]
        list1=list(np.linspace(min(bins),max(bins),100))
        list1_conv = [list1_val *int(unit) for list1_val in list1]
        popt, pcov = curve_fit(self.gaus,bins, n , p0 = [a_initial, mean_initial, sigma_initial])

        plt.errorbar(bins_conv, n, yerr= nerror, fmt ='o', label = 'Histogram for ' +str(directoryname))
        plt.plot(list1_conv,  self.gaus(list1, *popt), 'k', label = 'Gaussian Fit for ' +str(directoryname))
        plt.xlabel('Jitter Value [microvolt]')
        plt.ylabel('Number of Pulses')
        plt.legend()
        plt.show()

   #simulate one pulse
    def simulate_par(self, xmin_s=0, xmax_s=1000, points = 1000, mpvmaxlim =400, Amaxlim= 1, ftype = 'rand', plotting = False):
        x_s= np.array(np.linspace(xmin_s, xmax_s, points))
        global jitter, mpv, eta, A
        if ftype == 'rand':
            jitter, mpv, eta, A = self.par_normal(mpvmaxlim, Amaxlim)
        elif ftype == 'uniform':
            jitter, mpv, eta, A = self.par_uniform(mpvmaxlim, Amaxlim)
        y = pylandau.landau(x_s, mpv, eta, A)
        yj = [value + np.random.normal(0, np.sqrt(jitter)) for value in y]
        plt.plot(x_s, yj)
        if plotting == True:
            plt.show()
        return (jitter, mpv, eta, A, y, yj, mpvmaxlim, Amaxlim, x_s)
    
   #simulate multiple pulses
    def simulate_multiplepulses(self, xmin_s=0, xmax_s=1000, points = 1000,mpvmaxlim = 400, Amaxlim= 1, n_pulses = 3):
        ylist = [] 
        yflist = []
        jitter, mpv, eta, A, y, yj, mpvmaxlim, Amaxlim, x_s = self.simulate_par()
        print(mpv,eta, A)
        ylist.append(y)
        
        for n in range(n_pulses -1):
            jitter2, mpv2, eta2, A2, y2, yj2, mpvmaxlim2, Amaxlim2, x_s2 = self.simulate_par(mpvmaxlim  =mpv, Amaxlim = A, ftype = 'uniform')
            ylist.append(y2)
            print(mpv2, eta2, A2)
        yf = 0
        for i in range(len(y)):
            for n in range(n_pulses):
                yf += ylist[n][i]
            yflist.append(yf)
        plt.plot(x_s, yflist)
        plt.show()

