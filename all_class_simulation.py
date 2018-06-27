#!/usr/bin/env python
import sys, os, pylandau, all_class
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import product
from random import choice, uniform
#-------------------------------------------------------------------------------
class Simulate():
    def __init__(self, jitterfile, mpvfile, etafile, Afile, n_pulses=1 ):
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
            
   #xvalues and yvalues are lists
    #def number_pulses(self, data):
        #par = all_class.ScopeTrace(data).parameters()
        #newy = [xval- 
        
        
