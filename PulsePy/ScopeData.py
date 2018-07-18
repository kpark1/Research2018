#!/usr/bin/env python

import ScopeTrace      
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
class ScopeData():
	"""
	This is a class to manage a set of ScopeTrace objects.
	"""
	def __init__(self, trace_folder_dir, param_dir=None):
		'''
		Initializes a ScopeData object.

		:param str trace_folder_dir: Directory to folder containing traces to be analyzed.
		:param str self.dir: Directory for traces.
		:param str/None/optional self.param_dir: Directory for the parameter list.
		'''
		self.dir = trace_folder_dir
		self.param_dir = param_dir



	def data_read(self, filename):
		'''
		Returns ScopeTrace object from filename.
		:param str filename: String of filename. 

		'''
		with open(self.dir + filename, "r") as file1: 
			data = file1.read()
			return data




	def save_parameters(self, output_dir=None, filename=None, plotting=False):
		'''
		Saves parameters. ::
		

                  #Type in the name of the directory where the data files are stored 
		  import os, ScopeData
		  directory = <PATH> #e.g. "/home/kpark1/Work/SLab/data/" 
		  for file in sorted(os.listdir(directory)):
		      f = ScopeData.ScopeData(directory)
		      f.data_read(file)
		      f.save_parameters(plotting = False)

		:param str/None/optional output_dir: Directory for storing Landau fit parameters: mpv, eta, amp, and jitter (variance). The default directory is working directory.
		:param str/None/optional filename: Name of a new saved csv files that ends with '.csv'. If None, then the function creates a filename based on the trace folder title. 
		:param bool/optional plotting: If True, it plots each fitted curve. If False, it does not generate any graphs.
		'''

		#hack to prevent unwanted messages printing
		class NullWriter(object):
			def write(self, arg):
				pass
		nullwrite = NullWriter()
		oldstdout = sys.stdout

		#initialize variables
		count = 0
		landau_param_list = []
		zero_time = time.time()
		folder_size = len(os.listdir(self.dir))

		#Loops through files
		for curr_file in sorted(os.listdir(self.dir)):

			#Prints progress
			count += 1
			if plotting:
				print(str(count) + ' of ' + str(folder_size))
			elif time.time() - zero_time > 2:
				percent_done = round(float(count)/float(folder_size) * 100, 2)
				print('Progress: ' + str(percent_done) + '%')
				zero_time = time.time()

			with open(self.dir + curr_file, "r") as file1: 
				data = file1.read()
				trace = ScopeTrace.ScopeTrace(data)

			#store parameters
			try:
				baseline, jitter = trace.find_baseline_and_jitter(trace.get_xmin(), trace.get_trigger_point())
			except:
				baseline, jitter = trace.find_baseline_and_jitter(trace.get_xmin(), trace.get_xmin() + (trace.get_xmax() - trace.get_xmin())/10)
			parameters = trace.parameters()
			landau_param_list.append([str(curr_file)] + parameters + [jitter])

			#plotting
			if plotting:
				trace.plot(fit_param=parameters)
				plt.title(str(curr_file))
				plt.legend()
				plt.show()

		#saving
		if output_dir == None:   #save to working directory if none specified
			output_dir = str(os.getcwd())
			print(output_dir)
		if filename == None:	#generate filename if not specified
			filename = self.dir.split('/')[-2] + '_parameters'
		self.param_dir = output_dir + '/' + filename	#saves location of parameters
		savefile = open(self.param_dir, 'w')
		with savefile:
			writer = csv.writer(savefile)
			writer.writerow(['Filename', 'MPV', 'Eta', 'Amp', 'Jitter(Variance)'])
			writer.writerows(landau_param_list)
		






	def search_pulses(self, conditions, parameters, and_or='and', plotting=True):
		'''
		Returns a list of files that satisfy conditions from a user input with an option of plotting the pulses.
		Requires a directory where parameters have already been saved.:: 
      

                  #Type in the name of the directory where the data files are stored and the output directory
		  import ScopeData
		  directory =  <PATH> #e.g. "/home/kpark1/Work/SLab/data/"
		  f = ScopeData.ScopeData(directory)
		  f.save_parameters()
		  print(f.search_pulses([lambda x: x < .002, lambda x: x < .004],
                                              ['amp', 'mpv'], plotting = False))

	     

		:param list conditions: List of boolean functions. 
		:param list parameters: List of parameters [mpv, eta, amp] to check if the conditions apply to them. The list must have the same length as conditions. 
		:param str/optional and_or: String of either 'and' or 'or'. If the input is 'and', the method returns files that meet all of the given conditions. If the input is 'or', it returns files that meet any of the conditions. 
		:param bool/optional plotting: If True, it plots the pulses from the data.:: 
		'''
		starred_files = []
		param_dict = {'mpv': 1, 'eta': 2, 'amp': 3, 'jitter': 4}  #maps str parameter input to location in list

		#loop through csv files
		with open(self.param_dir, 'r') as savefile:
			reader = csv.reader(savefile)
			firstline = True
			for row in reader:	#goes through each file's parameters
				if firstline:	#checks if firstline and skips
					firstline = False
					continue
				else:
					if and_or == 'and':
						meets_conditions = True
						i = 0
						while meets_conditions and i < len(conditions):  #goes through each condition if all have
							meets_conditions = conditions[i](float(row[param_dict[parameters[i]]])) #been met so far
							i += 1

					elif and_or == 'or':
						meets_conditions = False
						for i in range(len(conditions)):   #checks for any met condition
							if conditions[i](float(row[param_dict[parameters[i]]])):
								meets_conditions = True
								break
					else:
						raise ValueError('Cannot read and/or input')

					if meets_conditions:
						data_file_dir = self.dir + row[0]
						starred_files.append(data_file_dir)

						#plotting
						if plotting:
							#initial settings
							with open(data_file_dir, "r") as data_file:
								data = data_file.read()
								trace = ScopeTrace(data)
								trace.plot([float(row[1]), float(row[2]), float(row[3])])
								plt.title(row[0])
								plt.show()

		return starred_files


	def histogram(self, parameter, hbins=10, hrange=None, hcolor= 'r', hedgecolor = 'k', halpha = .5):
		'''
		Makes a histogram of parameters.
		Returns a list parameters, a mean value and standard deviation, of Gaussian fit to histogram if parameter == 'eta' or 'jitter'::

		  import ScopeData
		  directory = <PATH> "/home/$USERNAME/Work/SLab/data/"
		  f = ScopeData.ScopeData(directory)
		  f.histogram('jitter')
		  plt.show()

		:param string parameter: Name of parameters among jitter, eta, mpv, and  amp.
		:param integer/optional hbins: Number of bins.
		:param list/optional hrange: Histogram Range 
		:param string/optional hcolor: Color of histogram bins
		:param string/optional hedgecolor: Color of edges of the bins
		:param float/optional halpha: Level of transparency in color of bins
		'''
		jitter, mpv, eta, amp = [], [], [], []
		with open(self.param_dir, "r") as savefile: 
			reader = csv.reader(savefile)
			firstline = True
			for row in reader:	#goes through each file's parameters
				if firstline:	#checks if firstline and skips
					firstline = False
					continue
				else:
					mpv.append(float(row[1]))
					eta.append(float(row[2]))
					amp.append(float(row[3]))
					jitter.append(float(row[4]))

		if parameter == 'jitter':
			param_list = jitter
		elif parameter == 'eta':
			param_list = eta
		elif parameter == 'mpv':
			param_list = mpv
		elif parameter == 'amp':
			param_list = amp
		else:
			raise ValueError("Parameter should be 'jitter', 'eta', 'mpv', or 'amp'!")

		n, bins, patches = plt.hist(param_list, bins=hbins, range=hrange, color=hcolor, edgecolor=hedgecolor)

		bin_avg = []
		for i in range(len(bins) - 1):
			bin_avg.append((bins[i] + bins[i+1])/2)
		bin_avg = np.array(bin_avg)

		nerror = []
		for nval in n:
			nerror.append(float(np.sqrt(nval)))
		nerror = np.array(nerror)

		plt.errorbar(bin_avg, n, nerror, fmt ='o', label = 'Histogram for ' + str(self.dir.split('/')[-2]))

		if parameter == 'eta' or parameter == 'jitter':
			a_initial = max(n)

			#gets x values at peaks
			n_array = np.array(n)
			idbin = np.where(n_array == n_array.max())
			#if multiple x values of the same max y values, selects the first max
			idbin = idbin[0][0]
			mean_initial = bin_avg[idbin]

			sigma_initial = fwhm(bin_avg, n)
		
			#Gaussian fit
			list1 = list(np.linspace(min(bin_avg), max(bin_avg), 100))
			popt, pcov = curve_fit(gaus, bin_avg, n, p0 = [a_initial, mean_initial, sigma_initial])

			plt.plot(list1, gaus(list1, *popt), 'k', label = 'Gaussian Fit for ' + str(self.dir.split('/')[-2]))


		plt.xlabel(parameter + ' Value')
		plt.ylabel('Number of Events')
		plt.legend()

		if parameter == 'eta' or parameter == 'jitter':
			return [float(popt[1]), float(popt[2])]

