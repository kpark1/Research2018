#!/usr/bin/env python
'''PulsePy is a tool designed to facilitate the analysis of pulses. This package can create a Landau curve that best fits existing pulses and simulate new pulses. PulsePy is divided into two classes and a function: ScopeTrace, ScopeData, and simulate_pulses. 
ScopeTrace can identify x and y values from a CSV file and calculate baseline and jitter values, which correspond to the mean value and variance. With those values, the class can also convert and plot y values to facilitate the visualization of both the observed data and Landau fit curve. ScopeTrace also create the best fit of the observed pulse in the Landau distribution function using the curvefit function in pylandau package with three parameters: mpv, eta, and A. These three parameters each depends on the the x value of the peak, width, and amplitude of the pulse. The quality of a fitted function is determined based on the proximity of initial guess parameters to actual working parameters that have decent fits.

ScopeData allows users to store parameters and search for pulses that meet specific requirements. The function simulate_pulses allows users to simulate pulses with customized conditions provided by users.

To run this package, the following packages should be installed:
sys, os, pylandau, numpy, matplotlib, scipy, itertools, random, time, csv

'''
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
		:param {'f', 'i'}/optional type: Type of values evaluated. 'f' for float and 'i' for integer or index.
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
		:param str filename: String of filename. ::

		
		  directory = "/home/kpark1/Work/SLab/data/"
		  for file in sorted(os.listdir(directory)):
		      f = PulsePy.ScopeData(directory)
		      f.data_read(file)
		'''
		with open(self.dir + filename, "r") as file1: 
			data = file1.read()
			return data




	def save_parameters(self, output_dir=None, filename=None, plotting=False):
		'''
		Saves parameters. ::
		

		  directory = "/home/kpark1/Work/SLab/data_4/"
		  d=PulsePy.ScopeData(directory)
		  d.save_parameters(plotting = False)

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
				trace = ScopeTrace(data)

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
		Returns a list of files that satisfy conditions from a user input with an option of plotting the pulses. :: 


		     data  = PulsePy.ScopeData('/home/kpark1/SLab/data/', '/home/kpark1/SLab/savedata/data_1_parameters')
		     print(data.search_pulses([lambda x: x < .002, lambda x: x < .004], ['amp', 'mpv'], plotting = False))
	     

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


		  directory = "/home/kpark1/Work/SLab/data_4/"
		  d=PulsePy.Scope(directory)
		  d.histogram('jitter')
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


def fwhm(x, y):
	
	
		x_array = np.array(x)
		y_array = np.array(y)
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


def gaus(x, a, x0, sigma):
	'''
	Defines a gaussian function.

	:param list x: List of values.
	:param float a: Amplitude of the function.
	:param float x0: Expected value.
	:param float sigma: Sigma value.
	'''
	return a* np.exp(-.5*((x-x0)/sigma)**2)



def simulate_pulses(num_events, time_range, eta_stats, amp_stats, jitter_stats, trigger_threshold=None, baseline=0.0, trigger_offset=None, num_pulses=None, possion_parameter=1, plotting=False, plot_pulse=False, save=True, output_dir=None):
	'''
	Simulates Landau pulses with noise or jitter. 
	Returns ScopeData object. ::cd


	  PulsePy.simulate_pulses(2, np.linspace(0, 2e-06, 2500), [5e-08,0] ,[1e-01,0],[2e-06, 0], plotting = True, plot_pulse = True)
	  plt.show()

	:param integer num_events: Number of files of events to create. 
	:param array time_range: Time range (x axis) for simulation. 
	:param list eta_stats: List containing a mean value and a standard deviation of eta values of pulses: ([mean value, std dev]).
	:param list jitter_stats: List containing a mean value and a standard deviation of jitter (variance) of data: ([mean value, std dev]).
	:param list amp_stats: List containing lower and upper bounds of amplitude over a random distribution: ([min, max]).
	:param bool/optional trigger_threshold: Simulates an oscilloscope trigger threshold. The first pulse of which amplitude is equal to or greater than the trigger threshold will be found at the trigger offset. If None, it simulates a random scope window.						 
	:param float/optional baseline: Sets a baseline voltage.
	:param float/optional trigger_offset: X value of a triggered spot; If trigger_offset == None, the default trigger offset is 1/10 of the time range.
	:param integer/optional num_pulses: Number of pulses per event. If None, the number is picked randomly from poisson distribution. 
	:param float/optional possion_parameter: Number of pulses randomly picked based on Possion Distribution.
	:param bool/optional plotting: If True, it plots the simulated pulse. 
	:param bool/optional plot_pulse: If True, it plots landau pulses.
	:param bool/optional save: If True, it saves the pulse simulation in the output directory. 
	:param str/optional output_dir: Directory to a folder for the saved csv files. If None, it saves the csv files in a newly created folder in working directory.
	'''
	plt.gcf().clear()
	#find important initial values
	random_num_pulses = num_pulses == None
	if trigger_offset == None:
		trigger_offset = (time_range[-1] - time_range[0])/10 + time_range[0]
	sample_interval = time_range[1] - time_range[0]
	if sample_interval < 0:
		raise ValueError('Domain error; Should be increasing')

	for event in range(num_events):
		#initialize baseline
		xvals = np.array(time_range)
		yvals = np.linspace(baseline, baseline, len(xvals))

		#initialize mpv, eta and amp list for pulses
		pulse_mpv_list = []
		pulse_eta_list = []
		pulse_amp_list = []

		#get jitter for an event
		jitter = np.random.normal(jitter_stats[0], jitter_stats[1])

		#generate pulses
		if trigger_threshold == None:   #no trigger
			#get num of pulses
			if random_num_pulses:
				num_pulses = np.random.poisson(possion_parameter)
			
				
			#generate pulses
			for pulse in range(num_pulses):
				pulse_mpv_list.append(np.random.uniform(min(xvals), max(xvals)))
				pulse_eta_list.append(np.random.normal(*eta_stats))
				pulse_amp_list.append(np.random.uniform(*amp_stats))

		elif trigger_threshold >= baseline:
			raise ValueError('Please set trigger threshold below baseline!')

		elif abs(trigger_threshold - baseline) > amp_stats[1] * 0.99:
			raise ValueError('Trigger threshold too far away from baseline!')

		elif num_pulses == 0:
			raise ValueError('Set trigger with no pulses!')

		else:	#yes trigger
			good_event = False
			while not good_event:
				#get num of pulses
				if random_num_pulses:
					num_pulses = np.random.poisson(possion_parameter)

				#generate pulses
				if num_pulses > 0:
					#make first pulse
					pulse_mpv_list.append(float(trigger_offset))
					pulse_eta_list.append(np.random.normal(*eta_stats))
					if amp_stats == None:
						pulse_amp_list.append(np.random.uniform(baseline - trigger_threshold, 100 * np.sqrt(jitter)))
					else:
						pulse_amp_list.append(np.random.uniform(baseline - trigger_threshold, amp_stats[1]))

					#adjust mpv for trigger (offset != mpv) using a points on landau pulse

					#find x where pulse is at trigger
					y_guess = pylandau.landau(xvals/sample_interval, pulse_mpv_list[-1]/sample_interval, pulse_eta_list[-1]/sample_interval, pulse_amp_list[-1])  #by index (divide by sample interval) to avoid small eta error

					for i in range(len(y_guess)):
						if y_guess[i] >= abs(trigger_threshold - baseline):
							x0 = xvals[i]
							break
					#find delta x correction
					delta_x = pulse_mpv_list[-1] - x0
				#new mpv
					del pulse_mpv_list[-1]
					pulse_mpv_list.append(np.float32(float(trigger_offset) + delta_x))
					
					if num_pulses > 1:
						for pulse in range(2, num_pulses + 1):	#make rest of pulses
							#stats for next pulse
							pulse_mpv_list.append(np.random.uniform(min(xvals), max(xvals)))
							pulse_eta_list.append(np.random.normal(*eta_stats))
							if amp_stats == None:
								pulse_amp_list.append(np.random.uniform(np.sqrt(jitter), 100 * np.sqrt(jitter)))
							else:
								pulse_amp_list.append(np.random.uniform(*amp_stats))

							#check if good event (bad if high amplitudes before trigger)
							if pulse_mpv_list[-1] < trigger_offset and pulse_amp_list[-1] >= baseline - trigger_threshold:
								#erase everything and reset event
								pulse_mpv_list = []
								pulse_eta_list = []
								pulse_amp_list = []
								break
							elif pulse == num_pulses:	#marks good event if passes last loop
								good_event = True
					else:	#one pulse scenario
						good_event = True
		print(num_pulses)
		#create Landaus
		all_parameters = zip(pulse_mpv_list, pulse_eta_list, pulse_amp_list)
		for parameters in all_parameters:
			yvals = np.add(yvals, -1*pylandau.landau(xvals/sample_interval, parameters[0]/sample_interval, parameters[1]/sample_interval, parameters[2]))  #have to strech out axis to avoid small eta errors
			
		#add noise
		for j in range(len(yvals)):
			yvals[j] = np.random.normal(yvals[j], np.sqrt(jitter))
			
		if plotting:
			#plotting
			plt.plot(xvals, yvals, label='Simulated Data')
			print(len(yvals))
			if plot_pulse:
				#plotting landau pulses
				for parameters in all_parameters:
					y_pulse = -1*pylandau.landau(xvals/sample_interval, parameters[0]/sample_interval, parameters[1]/sample_interval, parameters[2])	#evaluating by index to avoid small eta errors
					y_pulse = np.add(y_pulse, np.array([baseline]*len(y_pulse)))
					plt.plot(xvals, y_pulse, label='Pulse')
			if trigger_threshold != None:
				plt.axvline(x=trigger_offset, label='Trigger Offset')
				plt.plot(xvals, np.linspace(trigger_threshold, trigger_threshold, len(xvals)), label='Trigger Threshold')
			plt.legend(loc = 'lower right')
			
		
		if save:
			#for csv file format need row with info

			#first row (Labels)
			empty_row_1 = ['Simulated Data', '', 'Trigger Threshold', 'Trigger Offset', 'Baseline', 'Start Time', 'End Time', 'Sample Interval', 'Jitter (Variance)', 'Number of Pulses']
			for pulse in range(1, num_pulses + 1):
				empty_row_1 = empty_row_1 + ['Pulse ' + str(pulse) + ' MPV', 'Pulse ' + str(pulse) + ' Eta', 'Pulse ' + str(pulse) + ' Amp']
			empty_row_1 = empty_row_1 + ([''] * (len(xvals) - len(empty_row_1)))
			empty_row_1 = np.array(empty_row_1)

			#second row (Values)
			empty_row_2 = ['', '', trigger_threshold, trigger_offset, baseline, xvals[0], xvals[-1], xvals[1] - xvals[0], jitter, num_pulses]
			for pulse in range(1, num_pulses + 1):
				empty_row_2 = empty_row_2 + [pulse_mpv_list[pulse - 1], pulse_eta_list[pulse - 1], pulse_amp_list[pulse - 1]]
			empty_row_2 = empty_row_2 + ([''] * (len(xvals) - len(empty_row_2)))
			empty_row_2 = np.array(empty_row_2)

			#third row (empty)
			empty_row_3 = [''] * len(xvals)
			empty_row_3 = np.array(empty_row_3)

			#saving
			if output_dir == None:   #save to working directory if none specified
				cwd = str(os.getcwd())
				created_folder = False
				count = 1
				while not created_folder:
					new_dir = os.path.join(cwd, 'sim_data_' + str(count) + '/')
					if not os.path.exists(new_dir):
						output_dir = new_dir
						created_folder = True
					count += 1
			if not os.path.exists(output_dir):    #creates directory if missing
				try:
					os.makedirs(output_dir)
				except OSError:
					pass
			savefile = open(output_dir + str(event) + '.csv', 'w')
			with savefile:
				writer = csv.writer(savefile)
				writer.writerows(zip(empty_row_1, empty_row_2, empty_row_3, xvals, yvals))

	return ScopeData(output_dir)


