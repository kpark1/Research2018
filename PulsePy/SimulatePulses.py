#!/usr/bin/env python

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
def fwhm(x, y):
	'''
	Finds an approximate full width half maximum.
	:param list x: List of x values.
	:param list y: List of y values.
	'''
	
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



def simulate_pulses(num_events=2, time_range=np.linspace(0, 2e-06, 2500), eta_stats=[5e-08,0], amp_stats=[1e-01,0], jitter_stats=[2e-06, 0], trigger_threshold=None, baseline=0.0, trigger_offset=None, num_pulses=None, possion_parameter=1, plotting=False, plot_pulse=False, save=True, output_dir=None):
	'''
	Simulates Landau pulses with noise or jitter. 
	Returns ScopeData object. ::


	 import SimulatePulses
	 SimulatePulses.simulate_pulses(2, np.linspace(0, 2e-06, 2500), 
	                          [5e-08,0] ,[1e-01,0],[2e-06, 0], 
                                  plotting = True, plot_pulse = True)
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



