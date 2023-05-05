#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 18 10:44:54 2023

@author: dmoreno

Script to perform a two-phase linear regression (Salow et al., 1987).
The idea is to calculate where an equilibrium value is reached in a simulation.

Loosely speaking: we calculate two coupled linear regresions and calculate the
estimates (slope and zero value) that minimizes the residual sum of squares.

Two-phase linear regression:

y_i = a_0 + b_0 * i + b * (i - c) * ind_c

where b = b_1 - b_0, ind_c = 0 if i <= 0 and ind_c = 1 if i> 0.

Regressor variables:
--> i
--> (i - c) * ind_c 

Estimates:
--> a_0
--> b_0
--> b

For a fixed value of c, T_i is a normal linear regression. The estimates
are found from fitting the model with a fixed value of c. To find c,
we search over possible values to find the one that minimizes the residual sum of squares.
Hinkley (1971) gives a efficient search algorithm, though for small datasets a simple
grid search is reasonable.

Python documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

"""

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from netCDF4 import Dataset
from dimarray import get_datadir
import matplotlib.pyplot as plt
    

# Directories.
path_fig   = '/home/dmoren07/figures/v1.751/paper.a/LIS-16KM/t.eq/'
path_yelmo = '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/t.eq/'
file_name  = 'yelmo1D.nc'

# Options.
save_fig        = False
plot_regression = True
plot_r_sq       = True

# List subdirectories in ensemble and order.
ensemble = os.listdir(path_yelmo)
ensemble.sort()
print("Ensemble = ", ensemble)

# Ensemble name.
ensemble_name = ['Linear',  
				 'Power',   	
				 'Coulomb', 
				 'Therm']


# Let us create a dictionary with variable names.
yelmo_name = ['time', 'V_sl']
var_name   = ['t', 'V_sl']

# Dimension.
l_var = len(var_name)


# Number of independent variables.
n_var = 2


# Function that performs a two-pahase linear regression.

def two_phs_regress(path_yelmo, file_name, yelmo_name, var_name, n_var):

	# Current path to data.
	path_now = path_yelmo+str(file_name)

	# Open nc file in read mode.
	nc_SSA = os.path.join(get_datadir(), path_now)
	data   = Dataset(nc_SSA, mode='r')

	# Load data from yelmo1D.nc. t_n can be an array, np.shape(x) = (len(t_n), y, x)
	# Access the globals() dictionary to introduce new variables.
	for i in range(l_var):
		globals()[var_name[i]] = data.variables[yelmo_name[i]][:]


	# Dimensions.
	l = len(t)

	# Vector with integers of dimension.
	k = np.linspace(0, l, l, dtype=int)

	# Changepoint index ranging from c_0 to c_f.
	c_0 = 10
	c_f = l-1
	c_s = np.linspace(c_0, c_f, c_f-c_0, dtype=int)
	l_c = len(c_s)

	# Allocate matrix with regressor variables (t and t_c).
	x = np.empty([l, n_var])
	x[:,0] = t

	# Prepare variables.
	r_sq = np.empty(l_c)
	rss  = np.empty(l_c)
	bias = np.empty(l_c)
	coef = np.empty([l_c, n_var])

	# Loop over possibe c values.
	for i in range(l_c):

		# Create time vector with zeros below current c index value.
		t_c = t.copy()
		t_c = np.where(k > c_s[i], t_c, 0.0)

		# Allocate matrix.
		x[:,1] = t_c

		# The next step is to create the regression model as 
		# an instance of LinearRegression and fit it with .fit()
		# The result of this statement is the variable model referring to the object 
		# of type LinearRegression. It represents the regression model fitted with existing data.
		model = LinearRegression().fit(x, V_sl)

		# Coefficient of determination 𝑅².
		r_sq[i] = model.score(x, V_sl)

		# Independent term in the linear model (holds the bias V_sl_0).
		bias[i] = model.intercept_

		# Estimated coefficients for the linear regression problem.
		coef[i,:] = model.coef_

		# Calculate residual sum of squares (RSS).
		f      = coef[i,0] * t + coef[i,1] * t_c + bias[i]
		rss[i] = np.sum( ( V_sl - f )**2 )


	# Find index corresponding to the max value of R² and RSS.
	# Ensure we do not take the very beginning of the time series.
	c_max_r_sq = c_s[np.argmax(r_sq)]
	c_max_rss  = c_s[np.argmin(rss)]

	# Linear regression (univariate) of data points where i > c_max to show that 
	# the slope after this point is ~0.
	# Create vectors.
	t_c_max = t[c_max_rss:(l-1)]
	V_c_max = V_sl[c_max_rss:(l-1)]

	# Reshape training data to be a matrix (in case of 1-D input array).
	t_c_max = t_c_max.reshape(-1, 1)

	# Create model.
	model_c_max = LinearRegression().fit(t_c_max, V_c_max)

	# Coefficient of determination 𝑅².
	r_sq_c_max = model_c_max.score(t_c_max, V_c_max)

	# Independent term in the linear model (holds the bias V_sl_0).
	bias_c_max = model_c_max.intercept_

	# Estimated coefficients for the linear regression problem.
	coef_c_max = model_c_max.coef_

	# Line after c_max. the slope after this point must be ~0.
	y_c_max = coef_c_max[0] * t_c_max + bias_c_max

	out = [V_sl, t, coef, bias, r_sq, rss, c_max_rss, c_max_r_sq, c_s]

	return out



# Call two-phase linear regression for each time series.
sol_tot = []

# Loop over all runs of ensemble.
for i in ensemble:

	# Call function
	sol = two_phs_regress(path_yelmo, i, yelmo_name, \
						  var_name, n_var)

	# Append current regression.
	sol_tot.append(sol)

# Convert to array.
sol_tot = np.array(sol_tot)


# Variable plot names from solution.
var_plot = ['V_sl', 't', 'coef', 'bias', 'r_sq', 'rss', 'c_max_rss', 'c_max_r_sq', 'c_s']
l_plot   = len(var_plot)

# Name variables from solution output.
for i in range(l_plot):
	globals()[var_plot[i]] = sol_tot[:,i]


#################################################################
#################################################################

# PLOTS.

if plot_regression == True:

	# Figure.
	fig = plt.figure(dpi=600, figsize=(6,4.5))
	ax  = fig.add_subplot(111)

	# Use LaTeX.
	plt.rcParams['text.usetex'] = True

	color = ['blue', 'red', 'darkgreen', 'purple']
	label = [r'$ \mathrm{Linear} $', 
			 r'$ \mathrm{Power} $', 
			 r'$ \mathrm{Coulomb} $', 
			 r'$ \mathrm{Coulomb \ therm.} $']

	# Plot yelmo1D.nc time series. len(ensemble)
	for i in range(len(ensemble_name)):
		ax.plot(t[i], V_sl[i], linestyle='-', color=color[i], marker='None', \
				markersize=3.0, linewidth=3.0, alpha=1.0, label=label[i]) 
		
		"""
		ax.plot(np.full(100,t[i][c_max_rss[i]]), np.linspace(30,50,100), linestyle=':', \
					color=color[i], marker='None', markersize=3.0, \
						linewidth=1.5, alpha=1.0, label=r'$ t_c = \ $'+str(1e-3*t[i][c_max_rss[i]]))
		"""


	# Line after c_max point to show that slope must be ~0.
	#ax.plot(t_c_max, y_c_max, linestyle=':', color='red', marker='None', \
	#				markersize=3.0, linewidth=2.5, alpha=1.0, \
	#					label=r'$ \mathrm{Linear \ regression, \ } \tilde{c} = \ $'+str(c_max_rss))

	
	# Vertical line to show where c_max_rss is located.
	for i in range(len(ensemble)):
		ax.plot(np.full(100,t[i][c_max_rss[i]]), np.linspace(30,50,100), linestyle=':', \
					color=color[i], marker='None', markersize=3.0, \
						linewidth=1.5, alpha=1.0, label=r'$ t_c = \ $'+str(1e-3*t[i][c_max_rss[i]]))

	
	
	# Plot yelmo1D.mc time series. len(ensemble)
	#ax.plot(t, V_sl, linestyle='-', color='blue', marker='None', \
	#			markersize=3.0, linewidth=2.5, alpha=1.0, label=r'$ \mathrm{Yelmo} $') 
	
	plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.3), ncol = 2, frameon = True, \
			   framealpha = 1.0, fontsize = 13, fancybox = True)

	ax.set_ylabel(r'$ V_{\mathrm{sl}} \ (10^6 \ km^3)$', fontsize=18)
	ax.set_xlabel(r'$ \mathrm{Time} \ (kyr)$', fontsize=18)

	ax.set_xlim(-100.0, 1.5e5)
	ax.set_ylim(32.0, 40.0)

	ax.yaxis.label.set_color('black')

	
	ax.set_xticks([0, 25000, 50000, 75000, 100000, \
						125000, 150000])
	ax.set_xticklabels(['$0$', '$25$', '$50$', \
						'$75$', '$100$', '$125$', '$150$'], fontsize=13)

	ax.set_yticks([32, 34, 36, 38, 40])
	ax.set_yticklabels(['$32$', '$34$', '$36$', '$38$', '$40$'], fontsize=13)
	

	ax.tick_params(axis='both', which='major', length=4, colors='black')

	ax.grid(axis='y', which='major', alpha=0.85)

	plt.tight_layout()

	if save_fig == True:
		plt.savefig(path_fig+'V_sl_time_series_t_eq.png', bbox_inches='tight')

	# Display and close figure.
	plt.show()
	plt.close(fig)



if plot_r_sq == True:

	color = ['blue', 'red', 'darkgreen', 'purple']
	label = [r'$ \mathrm{Linear} $', 
			 r'$ \mathrm{Power} $', 
			 r'$ \mathrm{Coulomb} $', 
			 r'$ \mathrm{Coulomb thrm.} $']
	
	fig = plt.figure(dpi=600, figsize=(6,4))
	ax  = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	plt.rcParams['text.usetex'] = True

	for i in range(len(ensemble_name)):
		ax.plot(c_s[i], r_sq[i], linestyle='-', color=color[i], marker='None', \
				markersize=3.0, linewidth=2.5, alpha=1.0, label=label[i]) 
		ax2.plot(c_s[i], rss[i], linestyle='-', color=color[i], marker='None', \
				markersize=3.0, linewidth=2.5, alpha=1.0, label=label[i]) 



	# Vertical line to show where c_max_rss is located.
	for i in range(len(ensemble)):

		# Determination coeff.
		ax.plot(np.full(100,c_max_r_sq[i]), np.linspace(0.4,1.0,100), linestyle=':', \
					color=color[i], marker='None', markersize=3.0, \
						linewidth=1.5, alpha=1.0, label=r'$ t_c = \ $'+str(1e-3*t[i][c_max_rss[i]]))
		
		# Root mean square error.
		ax2.plot(np.full(100,c_max_rss[i]), np.linspace(25,125,100), linestyle=':', \
					color=color[i], marker='None', markersize=3.0, \
						linewidth=1.5, alpha=1.0, label=r'$ t_c = \ $'+str(1e-3*t[i][c_max_rss[i]]))

	#plt.legend(loc='best', ncol = 1, frameon = True, \
	#		framealpha = 1.0, fontsize = 12, fancybox = True)

	ax2.set_xlabel(r'$ t_c \ (\mathrm{kyr}) $', fontsize=18)
	ax.set_ylabel(r'$ R^2 $', fontsize=18)
	ax2.set_ylabel(r'$ \sum \left ( y_i - f(x_i) \right )^2 $', fontsize=18)

	ax.set_xlim(c_s[0][0], c_s[0][len(c_s[0])-1])
	ax2.set_xlim(c_s[0][0], c_s[0][len(c_s[0])-1])
	ax.set_ylim(0.4, 1.0)
	ax2.set_ylim(25, 125)

	ax.yaxis.label.set_color('black')

	ax.set_xticks([10, 30, 50, 70, 90, 110, 130, 150])
	ax.set_xticklabels([])

	ax2.set_xticks([10, 30, 50, 70, 90, 110, 130, 150])
	ax2.set_xticklabels(['$10$', '$30$', '$50$', \
							'$70$', '$90$', '$110$', 
							'$130$', '$150$'], fontsize=12)
	
	ax.set_yticks([0.4, 0.6, 0.8, 1.0])
	ax.set_yticklabels(['$0.4$', '$0.6$', '$0.8$', '$1.0$'], fontsize=12)
	
	ax2.set_yticks([25, 50, 75, 100, 125])
	ax2.set_yticklabels(['$25$', '$50$', '$75$', '$100$', '$125$'], fontsize=12)

	ax.tick_params(axis='x', which='major', length=0, colors='black')

	ax.grid(axis='x', which='major', alpha=0.85)
	ax2.grid(axis='x', which='major', alpha=0.85)

	plt.tight_layout()

	if save_fig == True:
		plt.savefig(path_fig+'root_mean_square_error.png', bbox_inches='tight')

	# Display and close figure.
	plt.show()
	plt.close(fig)