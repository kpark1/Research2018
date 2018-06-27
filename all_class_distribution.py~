#!/usr/bin/env python
import sys, os, pylandau, all_class
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import product
from random import choice
import csv

#-------------------------------------------------------------------------------
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
            par = all_class.ScopeTrace(data)
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
        
    
