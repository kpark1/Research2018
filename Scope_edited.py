import sys, os, pylandau
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import product
from decimal import Decimal
import time
import csv

class ScopeTrace:
	undefined_value = -9999999

	def __init__(self, data, n_average=1):
		# data record
		self.data = data
		self.n_average = n_average
		
		# get the scopes raw readings (average as many as you want)
		x = 0
		y = 0
		i = 0
		n = 0
		self.xvalues = []
		self.yvalues = []
		for line in data.split("\n"):
			f = line.split(',')
			if len(f)<5:
				x += float(i)
				y += float(f[4])
				n += 1
			if n>n_average:
				self.xvalues.append(x/n)
				self.yvalues.append(y/n)
				n = 0
				x = 0
				y = 0
			i += 1

	def find_baseline_and_jitter(self, xmin=0, trigger_offset=250):
		yval_bl = self.yvalues[xmin+1:trigger_offset-1]
		baseline = sum(y_val_bl)/len(y_val_bl)
		jitter = sum([y_j **2 for y_j in y_values_jitter]) -baseline**2
		return (baseline, jitter)

	#def find_number_of_pulses(self, baseline, threshold, delta_min):
		
	def inverted_y(self, fit_param=None, xmin=0, trigger_offset=250, plotting = False):
		# find baseline and jitter 
		baseline,jitter = find_baseline_and_jitter(xmin,trigger_offset)
		#inverted y values
		return -(val-baseline) for val in self.yvalues
		plt.plot(x, y, label = 'Data')
		if fit_parm != None and len(fit_param) ==3:
			plt.plot(x, pylandau.landau(x, *fit_param), label ='Landau Fit') 

class ScopeData:
	#Class to manage a set of ScopeTrace objects.
	def __init__(self, directory):
		self.dir = directory
	def save_parameters(self, output_dir, filename, plotting=False):
		'''
		data_folder: folder containing traces in csv format
		output_dir: directory to store parameters of Landau fits (mpv, eta, amplitude, jitter (variance))
		filename: name of new saved csv files (put in .csv format)
		plotting: will plot each fit
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

				#decode the scope trace
				trace = ScopeTrace(data)
				#find baseline and jitter 
				baseline, jitter = trace.find_baseline_and_jitter()
				#set x and y
				y = trace.inverted_y()
			
				#gets x values at peaks
				y_array = np.array(y)
				idx = np.where(y_array == y_array.max())

				#if multiple x values of the same max y values, selects the first max
				idx = idx[0][0]
				x_values_peak = x[idx]

				#Curvefit for Landau with parameters mpv,  eta, A 
				mpv = x_values_peak
				rmin= 1

				try: 
					sys.stdout = nullwrite #disable printing
					landau_par_rmin, pcov_rmin = curve_fit(pylandau.landau, x, y, p0= (mpv, rmin, rmin))
					array_1 = np.ndarray.tolist(np.around(landau_par_rmin, decimals = 3))
					param_list = [array_1]
					workinglandaupar = [array_1]
					sys.stdout = oldstdout #enable printing

					diff_sq_fn = lambda parameter: round(sum((y-pylandau.landau(x, *parameter))**2), 4)
					initial_diff_sq = [diff_sq_fn(landau_par_rmin)]
				
				except RuntimeError as message:
					print(str(message) + ' for ' + str(curr_file))
					continue
				except TypeError as message:
					print(str(message) + ' for ' + str(curr_file))
					continue

				#only one best possible parameter with the least diff_sq
				for eta, A in product(np.linspace(rmin, 25, 5), np.linspace(rmin, 25, 5)):
					try:
						sys.stdout = nullwrite #disable printing
						landau_par, pcov = curve_fit(pylandau.landau, x, y, p0= (mpv, eta,A))
						landau_par = np.ndarray.tolist(np.around(landau_par, decimals =3))
						diff= diff_sq_fn(landau_par)
						par = param_list[0]
						sys.stdout = oldstdout #enable printing

						if initial_diff_sq[0] < .01:
							break
						elif landau_par != par and diff < initial_diff_sq[0]:
							param_list.append(landau_par)
							workinglandaupar[0]=landau_par
							initial_diff_sq[0]= diff
							break
					except RuntimeError as message:
						print(str(message)+ str(curr_file))
						continue
					except TypeError as message:
						print(str(message) +'2nd'+ str(curr_file))
						continue

			#store parameters
			landau_param_list.append([str(curr_file)] + workinglandaupar[0] + [jitter])

			#plotting
			if plotting:
				trace.plot()
				plt.plot(x, pylandau.landau(x, *workinglandaupar[0]), label='Landau Fit')
				plt.title(str(curr_file))
				plt.legend()
				plt.show()

		#saving
		savefile = open(output_dir + filename, 'w')
		with savefile:
			writer = csv.writer(savefile)
			writer.writerow(['Filename', 'MPV', 'Eta', 'Amplitude', 'Jitter(Variance)'])
			writer.writerows(landau_param_list)

class Histogram:
	def __init__(self, data):
		self.data = data
		
		
