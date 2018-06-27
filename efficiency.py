#!/usr/bin/env python
import sys, os, ScopeTrace, pylandau, all_class, all_class_simulation, Scope2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.optimize import curve_fit
from itertools import product
import csv
#------------------------------------------------------------------------------
class Efficiency():
	'''
	This is a class to manage a scope trace.
	'''
	
	undefined_value = None

	def __init__(self, data, n_average = 1, efficiency = True):
		self.data = data
		self.n_average = n_average
		self.xvalues = []
		self.yvalues = []
		if efficiency == True:
			self.zvalues = []
			self.tvalues = []
                #??????look below
		self.sample_interval = 1
		
	
		x = 0
		y = 0
	        n = 0
		i = 0
		if efficiency == True:
			z=0
		
		for line in data.split('\n'):	#get scope reading with average
			f = line.split(',')
			try:
				x += float(f[3])
				y += float(f[4])
				z+=float(f[1])
				if efficiency == True:
					t = f[0]
				n += 1
				if n >= n_average:
					self.xvalues.append(x/n)
					self.yvalues.append(y/n)
					if efficiency == True:
						self.zvalues.append(z)
						self.tvalues.append(t)
					n = 0
					x = 0
					y = 0
					if efficiency == True:	
						z=0
				i += 1
	
			except:
				pass
		print(self.xvalues)
		print(self.yvalues)
		if efficiency == True:	
			self.tvalues=np.asarray(self.tvalues)
			idx_num_pulses= np.where(self.tvalues == 'Number of Pulses')[0][0]
			self.num_pulses = self.zvalues[idx_num_pulses]
			idx_trigger_point = np.where(self.tvalues == 'Trigger Offset')[0][0]
			self.trigger_point = self.zvalues[idx_trigger_point]
			#print(self.num_pulses, self.trigger_point)
			
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
	
	def get_xmin(self):
		'''
		Returns a minimum x value from the data.
		'''
		return self.xvalues[0]

	def inverted(self):	
	
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
	

	def parameters(self, yvals=None):
		
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
		A = y_array.max()
		eta = self.fwhm()
		landau_par, pcov_rmin = curve_fit(pylandau.landau, x, y, p0=(mpv, 1, A))
		sys.stdout = oldstdout #enable printing
		return [float(landau_par[0] * self.sample_interval), float(landau_par[1] * self.sample_interval), float(landau_par[2])]

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
		while curr_ymax > 1*std:
			landaufcn = pylandau.landau(x_index, current_fit[0]/self.sample_interval, current_fit[1]/self.sample_interval, current_fit[2])
			y = [-landaufcn[i] + y[i] for i in range(len(y))]
			curr_ymax = max(y)
			count += 1
			current_fit = self.parameters(y)
		return count

		
	def evaluate(self, filename):
		
		if Scope2.ScopeTrace(filename).get_num_pulses() == self.num_pulses:
			print('Match with '+ str(self.get_num_pulses())+':'+ str(self.num_pulses))
		elif Scope2.ScopeTrace.get_num_pulses() != self.num_pulses:
			print('Does not Match'+ str(self.get_num_pulses())+':'+ str(self.num_pulses))
		else:
			print('Check if there is any error!')
	

	def plot(self, fit_param=None):
		'''
		Plots trace data with optional Landau fit.

		:param array/None/optional fit_param: Array of Landau parameters (mpv eta amplitude).
		'''
		#set x and y
		x = np.array(self.xvalues)
		y = np.array(self.inverted())

		#plotting
		plt.plot(x, y, label='Data')
		if fit_param != None and len(fit_param) == 3:
			plt.plot(x, pylandau.landau(np.linspace(0, len(x) - 1, len(x)), fit_param[0]/self.sample_interval, fit_param[1]/self.sample_interval, fit_param[2]), label='Landau Fit')

