#!/usr/bin/env python
#ScopeTrace.py
'''PulsePy is a tool designed to facilitate the analysis of pulses. This package can create a Landau curve that best fits existing pulses and simulate new pulses. PulsePy is divided into two classes and a function: ScopeTrace, ScopeData, and simulate_pulses. 
ScopeTrace can identify x and y values from a CSV file and calculate baseline and jitter values, which correspond to the mean value and variance. With those values, the class can also convert and plot y values to facilitate the visualization of both the observed data and Landau fit curve. ScopeTrace also create the best fit of the observed pulse in the Landau distribution function using the curvefit function in pylandau package with three parameters: mpv, eta, and A. These three parameters each depends on the the x value of the peak, width, and amplitude of the pulse. The quality of a fitted function is determined based on the proximity of initial guess parameters to actual working parameters that have decent fits.

ScopeData allows users to store parameters and search for pulses that meet specific requirements. The function simulate_pulses allows users to simulate pulses with customized conditions provided by users.

To run this package, the following packages should be installed:
sys, os, pylandau, numpy, matplotlib, scipy, itertools, random, time, csv

To install these packages, type in 'pip install <package>' in linux command line.'''
    
    
import sys, os, pylandau
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from itertools import product
from random import choice
import numpy.polynomial.polynomial as poly
import time
import csv

#-------------------------------------------------------------------------------#
class ScopeTrace():
	'''
	This is a class to manage a oscilloscope trace.
	'''
	#AttributeError: ScopeTrace instance has no attribute 'split' -> when you didn't read the data correct, the program gets confused
	undefined_value = None

	def __init__(self, data, n_average = 1):
		self.data = data
		self.n_average = n_average
		self.xvalues = []
		self.yvalues = []
		try:
			self.sample_interval = self.find_value('Sample Interval', data)
		except:
			#pass
			self.sample_interval = self.xvalues[1] - self.xvalues[0]
		

		x = 0
		y = 0
		n = 0
		i = 0
		for line in data.split('\n'):	#get scope reading with average
			f = line.split(',')
			try:
				x += float(f[3])
				y += float(f[4])
				n += 1
				if n >= n_average:
					self.xvalues.append(x/n)
					self.yvalues.append(y/n)
					n = 0
					x = 0
					y = 0
				i += 1
			except:
				pass
		
		try:
			self.trigger_point = self.find_value('Trigger Point', data) * self.sample_interval + self.xvalues[0]
		except:
			try:
				self.trigger_point = self.find_value('Trigger Offset', data)
			except:
				self.trigger_point = None




	def get_trigger_point(self):
		'''
		Returns an x value where the pulse was triggered. 
		'''
		return self.trigger_point




	def get_xmin(self):
		'''
		Returns a minimum x value from the data.
		'''
		return self.xvalues[0]



	def get_xmax(self):
		'''
		Returns a maximum x value from the data.
		'''
		return self.xvalues[-1]



	def find_baseline_and_jitter(self, xmin, xmax):
		''' 
		Finds a baseline and a jitter value, which correspond to a mean value and variance.

		:param float xmin: X minimum value.
		:param float xmax: X maximum value.
		'''
		n = 0
		sum1 = 0
		sum2 = 0

		for x,y in zip(self.xvalues, self.yvalues):
			if x > xmin and x < xmax:
				n = n + 1
				sum1 = sum1 + y
				sum2 = sum2 + y*y

		baseline = 0
		jitter = 0
		if n>0:
			baseline = sum1/n
			jitter = sum2/n - baseline*baseline
		return baseline, jitter




	def find_value(self, name, data, type="f"):
		'''
		Returns the oscilloscope settings such as record length, sample interval, units, etc.

		:param str name: Name of value of interest, such as 'Trigger Point'.
		:param str data: Read CSV file. Compatible with ScopeData.data_read().
		:param 'f'/'i'/optional type: Type of values evaluated. 'f' for float and 'i' for integer or index.
		'''
		value = self.undefined_value
		for line in data.split("\n"):
			f = line.split(',')
			if f[0] == name:
				if type == 'f':
					value = float(f[1])
					#print " Value[%s]  %f (F)"%(name,value)
				elif type == 'i':
					value = int(f[1])
					#print " Value[%s]  %d (I)"%(name,value)
				else:
					value = f[1]
					#print " Value[%s]  %s (S)"%(name,value)
				break
		return value


	
	def inverted(self):	
		'''
		Returns a list of y values of which baseline has been reset to zero and then are reflected about the x-axis.
		'''
		baseline = self.find_baseline_and_jitter(self.get_xmin(), self.trigger_point)[0]
		return [-(val-baseline) for val in self.yvalues]




	def fwhm(self):
		'''
		Returns an approximated full width at half maximum.
		'''
		x_array = np.array(self.xvalues)
		y_array = np.array(self.inverted())
		idx = np.where(y_array == y_array.max())
		idx = idx[0][0]
		y_closest_to_hm = min(y_array, key=lambda x: abs(x-.5*max(y_array)))
		idx_hm_left = np.where(y_array == min(y_array[:idx], key=lambda x: abs(x-y_closest_to_hm)))
		idx_hm_right = np.where(y_array == min(y_array[idx:], key=lambda x: abs(x-y_closest_to_hm)))
		#print(idx_hm_left)
		x_hm_left = x_array[idx_hm_left[0][0]]
		x_hm_right = x_array[idx_hm_right[0][0]]
		fw =  abs(x_hm_left - x_hm_right)
		return fw



	def get_num_pulses(self):
		'''
		Returns the number of pulses of a file.
		'''
		x = np.array(self.xvalues)
		x_index = np.linspace(0, len(x) - 1, len(x))
		y = self.inverted()
		curr_ymax = max(y)
		std = float(np.sqrt(self.find_baseline_and_jitter(self.get_xmin(), self.trigger_point)[1]))
		count = 0
		current_fit = self.parameters()
		while curr_ymax > 5*std:
			landaufcn = pylandau.landau(x_index, current_fit[0]/self.sample_interval, current_fit[1]/self.sample_interval, current_fit[2])
			y = [-landaufcn[i] + y[i] for i in range(len(y))]
			curr_ymax = max(y)
			count += 1
			current_fit = self.parameters(y)
		return count


	def parameters(self, yvals=None):
		'''
		Suppresses warning messages that do not affect the results of a Landau fitting method.
		Finds parameters of a landau distribution fit.

		:param list/None yvals: List of parameters, mpv, eta, amp, for a landau fitting curve.
		'''
		#hack to prevent unwanted messages printing
		class NullWriter(object):
			def write(self, arg):
				pass
		nullwrite = NullWriter()
		oldstdout = sys.stdout
		sys.stdout = nullwrite #disable printing

		x = np.linspace(0, len(self.xvalues) - 1, len(self.xvalues)) #scales by index so small eta not a problem
		if yvals == None:
			y = self.inverted()
		else:
			y = yvals

		#gets x values at peaks
		y_array = np.array(y)
		idx = np.where(y_array==y_array.max())
		#if multiple x values of the same max y values, selects the first max
		idx = idx[0][0]
		x_values_peak = x[idx]

		mpv = x_values_peak
		amp = y_array.max()
		eta = self.fwhm()
		landau_par, pcov_rmin = curve_fit(pylandau.landau, x, y, p0=(mpv, 1, amp))
		sys.stdout = oldstdout #enable printing
		return [float(landau_par[0] * self.sample_interval), float(landau_par[1] * self.sample_interval), float(landau_par[2])]




	def plot(self, fit_param=None):
		'''
		Plots trace data with optional Landau fit.

		:param array/None/optional fit_param: Array of Landau parameters: mpv, eta, and amp.
		'''
		#set x and y
		x = np.array(self.xvalues)
		y = np.array(self.inverted())

		#plotting
		plt.plot(x, y, label='Data')
		if fit_param != None and len(fit_param) == 3:
			plt.plot(x, pylandau.landau(np.linspace(0, len(x) - 1, len(x)), fit_param[0]/self.sample_interval, fit_param[1]/self.sample_interval, fit_param[2]), label='Landau Fit')




	def plot_range(self, xmin, xmax):
		'''
		Plots trace with given ranges.

		:param integer xmin: Minimum x (by index).
		:param integer xmax: Maximum x (by index).
		'''
		#set x and y
		x = np.array(self.xvalues[xmin: xmax])
		y = np.array(self.inverted()[xmin:xmax])
		#plotting
		plt.plot(x, y, label='Data')
		


