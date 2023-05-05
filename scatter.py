#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:44:04 2021

@author: dmoreno
"""

from __future__ import division
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dimarray import read_nc, get_datadir
import os
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colors as colors
plt.style.use("seaborn-white")



# Python script to make scatter plot of phase space from
# yelmo simulations.
"""
Options:

tau_u_colour_H_ice  ---> tau_b vs u_b with a color scale given by H_ice.
tau_u_colour_H_w    ---> tau_b vs u_b with a color scale given by H_w.
tau_u_colour_lmbd   ---> tau_b vs u_b with a color scale given by bedrock scaling lambda.
tau_H_size_Q_b      ---> tau_b vs H_ice with marker size given by frictional heat Q_b.   

"""

# Select options:
tau_u_colour_H_ice = False
tau_u_colour_H_w   = False
tau_u_colour_lmbd  = True
tau_H_size_Q_b     = False
save_figures       = True

# /home/dmoren07/yelmo-model_v1.64/yelmox/output/cb.1.0.till/
# '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/N.eff/'
# '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/overburden/'
path     = '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/overburden/'
path_fig = '/home/dmoren07/figures/v1.751/paper.a/LIS-16KM/overburden/'


# List all subdirectories using os.listdir
ensemble = os.listdir(path)
ensemble.sort()

# Choose ensemble
index = 3 # 3
ensemble_name = ensemble[index]
plot   = np.array([index])
l_plot = len(plot)

print('')
print('Ensemble = ', ensemble_name)
print('')

path_save = path_fig+ensemble_name+'/'


isdir = os.path.isdir(path_save)
if isdir == False:
	os.makedirs(path_save)


# We load f_pmp for variable shape.
nc_YELMO2D  = os.path.join(get_datadir(), path+ensemble[index]+'/yelmo2D.nc')
YELMO2D     = Dataset(nc_YELMO2D, mode='r')
f_pmp       = YELMO2D.variables['f_pmp'][:]

# Variables shape.
s = np.shape(f_pmp)


####################################################
################# Plots several frames ###################

# Frames to plot.
n_0 = s[0] - 2
n_f = s[0] - 1
t_n = np.arange(n_0, n_f, 1, dtype=int)
l_t = len(t_n)

# Let us create a dictionary with variable names.
var_name   = ['u', 'H', 'f', 'fg', 'tau', 'H_w']
yelmo_name = ['uxy_b', 'H_ice', 'f_pmp', 'f_grnd', 'taub', 'H_w']
l_var      = len(var_name)

# NC object. 
nc_YELMO2D  = os.path.join(get_datadir(), path+ensemble_name+'/yelmo2D.nc')
YELMO2D     = Dataset(nc_YELMO2D, mode='r')

# Load data from yelmo2D.nc. t_n can be an array, np.shape(x) = (len(t_n), y, x)
# Access the globals() dictionary to introduce new variables.
for i in range(l_var):
	globals()[var_name[i]] = YELMO2D.variables[yelmo_name[i]][t_n,:,:]


# Units.
tau = 1.0e-3 * tau
H   = 1.0e-3 * H


#########################################################################
#########################################################################

# Tau_b vs u_b coloured by the ice thickness.

if tau_u_colour_H_ice == True:

	# Number of point in the colourbar.
	N = 200
	N_1 = np.int(0.4*N)
	N_2 = np.int(0.2*N)

	# Min/max values.
	#max_value = np.nanmax(H[0,:,:])
	#min_value = np.nanmin(H[0,:,:])
	max_value = 5.0
	min_value = 0.0
	lim_min   = min_value
	lim_med_1 = 0.4 * (max_value - min_value)
	lim_med_2 = 0.6 * (max_value - min_value)
	lim_max   = max_value - min_value

	# Colours for levels.
	colors1 = plt.cm.seismic(np.linspace(1.0, 0.75, N_1))
	colors2 = plt.cm.cool(np.linspace(0.45, 0.0, N_2))      
	colors3 = plt.cm.PiYG(np.linspace(0.7, 1.0, N_1))

	# Concatenate all colour levels.
	levels = np.concatenate([np.linspace(lim_min,lim_med_1,N_1+1), \
							np.linspace(lim_med_1,lim_med_2,N_2), \
							np.linspace(lim_med_2,lim_max,N_1)])

	colors = np.vstack((colors1,colors2,colors3))
	cmap_v, norm_v = mcolors.from_levels_and_colors(levels, colors)
	cmap_v.set_over(plt.cm.RdYlBu(0))

	
	# FIGURE.
	# Plot a figure per frame.
	for k in range(l_t):

		fig = plt.figure(dpi=400)
		ax = fig.add_subplot(111)

		plt.rcParams['text.usetex'] = True

		# Allocate plot variables.
		u_plot      = np.empty((s[1], s[2]))
		tau_plot    = np.empty((s[1], s[2]))
		u_plot[:]   = np.nan
		tau_plot[:] = np.nan

		# Loop over all points.
		for i in range(s[1]):
			for j in range(s[2]): 
				
				# Avoid values below a certain threshold.
				if u[k,i,j] > 1.5 and H[k,i,j] > 1.0:   # if u[0,i,j] < 1.5 or H[0,i,j] < 1.0
					u_plot[i,j]   = u[k,i,j]
					tau_plot[i,j] = tau[k,i,j]			       
		
		# Scatter plot, c gives the colour variable and s the size.
		plt.scatter(u_plot, tau_plot, c = H[k,:,:], s=1, cmap=cmap_v, norm=norm_v)

		# Colourbar settings.
		ticks = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
		cbar  = plt.colorbar(ticks = ticks)

		cbar.set_ticklabels(['$0.0$', '$1.0$', '$2.0$', '$3.0$', '$4.0$', '$5.0$'])
		cbar.ax.tick_params(labelsize=17)
		cbar.ax.set_ylabel('$H_{\mathrm{ice}}$ $(km)$', labelpad=12, fontsize=17)

		# x ticks.
		ax.set_xticks([0, 100, 200, 300, 400, 500])
		ax.set_xticklabels(['$0$', '$100$', '$200$', \
							'$300$', '$400$', '$500$'], fontsize=17)

		# y ticks.
		ax.set_yticks([0, 50, 100, 150, 200])
		ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], fontsize=17)	

		# Axis labels.
		ax.set_xlabel(r'$ u_{b} \ (m/yr)$',fontsize=17)
		ax.set_ylabel(r'$\tau_{b} \ (kPa)$',fontsize=17)

		# Axis limits.
		ax.set_xlim(0.0, 500)
		ax.set_ylim(0.0, 200) # 160

		#ax.set_title(title_name[ens_index],fontsize=14)

		# Axis ticks.
		ax.tick_params(axis='both', which='major', length=4, color='black')

		# Grid.
		ax.grid(which='both')
		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		##### Frame name ########
		# Start labelling on n_0.
		k_lab = k + n_0

		# Frame name to save figure.
		if k_lab < 10:
			frame = '00'+str(k_lab)
		elif k_lab > 9 and k_lab < 100:
			frame = '0'+str(k_lab)
		else:
			frame = str(k_lab)

		# Tight plot format.
		plt.tight_layout()

		# Save figures if desired.
		if save_figures == True:
			plt.savefig(path_save+ensemble_name+ \
						'_scatter_u_xy_b_tau_b_'+str(frame)+'.png', bbox_inches='tight')
		
		# Display and close figure.
		plt.show()
		plt.close(fig)



#########################################################################
#########################################################################

# Tau_b distribution as a function of ice thickness.

if tau_H_size_Q_b == True:

	# Legend ticks.
	legend_ticks = [0.25, 0.50, 0.75, 1.0]

	# Labels in legend.
	mylabels = [r'$ 0.25 $', \
				r'$ 0.50 $', \
				r'$ 0.75 $', \
				r'$ 1.00 $' ]

	# Maximum tau_b value to be considered (kPa).
	tau_max = 200.0

	# Marker size is a cubic (n = 3) function of Q/Q_max. Factor 200.0 for visualization.
	n      = 3
	factor = 200.0

	for k in range(l_t):

		print('k = ', k)

		# Define variables.
		u_dist   = []
		H_dist   = []
		tau_dist = []

		# Loop over all points.
		for i in range(s[1]):
			for j in range(s[2]): 
				
				# Avoid values above a certain threshold.
				if tau[k,i,j] < tau_max:   # Alex's comment.  H[0,i,j] < 1.0
					
					# Statistical distribution for desired points.
					u_dist.append(u[k,i,j])
					H_dist.append(H[k,i,j])
					tau_dist.append(tau[k,i,j])
					
		# Convert to array for math operations.
		u_dist   = np.array(u_dist)
		H_dist   = np.array(H_dist)
		tau_dist = np.array(tau_dist)

		# Frictional energy: Q = tau_b * u_b.
		Q     = tau_dist * u_dist
		Q_max = np.nanmax(Q)
		# User max limit for comparison among experiments.

		# Total frictional heat.
		Q_tot = np.sum(tau * u)
		#print('Q_tot = ', Q_tot)
		

		# FIGURE. 
		# Plot a figure per frame.
		fig = plt.figure(dpi=400)
		ax  = fig.add_subplot(111)
		
		# Latex interpreter.
		plt.rcParams['text.usetex'] = True

		# Marker size is a cubic (n = 3) function of Q/Q_max. Factor 200.0 for visualization.
		sizes  = factor * ( Q / Q_max )**n # **2

		# Scatter plot, c gives the colour variable.
		scatter = plt.scatter(H_dist, tau_dist, sizes, c="green", alpha=0.5, marker='o')

		# Dummy plot to make legend.
		c = 0
		for area in legend_ticks:
			plt.scatter([], [], c='green', alpha=0.3, s=factor*area**n, label=str(mylabels[c]))
			c = c + 1

		legend = ax.legend(loc = "upper right", title = r"$ Q / Q_{\mathrm{max}} $", \
						   title_fontsize = 15, frameon = True, framealpha = 1.0, \
						   fontsize = 13, fancybox = True)

		# x ticks.
		ax.set_xticks([0, 1, 2, 3, 4, 5])
		ax.set_xticklabels(['$0$', '$1$', '$2$', '$3$', '$4$', '$5$'], fontsize=17)

		# y ticks.
		ax.set_yticks([0, 50, 100, 150, 200])
		ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], fontsize=17)	

		# Axis limits.
		ax.set_xlim(0.0, 5)
		ax.set_ylim(0.0, 200)
		
		# Axis ticks.
		ax.tick_params(axis='both', which='major', length=4, color='black')

		# Axis labels.
		ax.set_xlabel(r'$ H_{\mathrm{ice}} \ (km)$', fontsize=17)
		ax.set_ylabel(r'$\tau_{b} \ (kPa)$', fontsize=17)

		# Grid.
		ax.grid(which='both')
		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		# Set tittle.
		#ax.set_title(r'$n \ = \ $'+str(k), fontsize=16)

		##### Frame name ########
		# Start labelling on n_0.
		k_lab = k + n_0

		# Frame name to save figure.
		if k_lab < 10:
			frame = '00'+str(k_lab)
		elif k_lab > 9 and k_lab < 100:
			frame = '0'+str(k_lab)
		else:
			frame = str(k_lab)

		# Plot tight format.
		plt.tight_layout()

		if save_figures == True:
			plt.savefig(path_save+ensemble_name+ \
						'_scatter_tau_b_H_ice_'+str(frame)+'.png', bbox_inches='tight')
		
		# Display and close figure.
		plt.show()
		plt.close(fig)

		#print('path_save = ', path_save)



#########################################################################
#########################################################################

# Tau_b vs u_b colour by the basal water layer thickness.

if tau_u_colour_H_w == True:

	# FIGURE.
	# Plot a figure per frame.
	for k in range(l_t):

		fig = plt.figure(dpi=400)
		ax = fig.add_subplot(111)

		plt.rcParams['text.usetex'] = True

		# Allocate plot variables.
		u_plot      = np.empty((s[1], s[2]))
		tau_plot    = np.empty((s[1], s[2]))
		u_plot[:]   = np.nan
		tau_plot[:] = np.nan

		# Loop over all points.
		for i in range(s[1]):
			for j in range(s[2]): 
				
				# Avoid values below a certain threshold.
				if u[k,i,j] > 1.5 and H[k,i,j] > 1.0:   # if u[0,i,j] < 1.5 or H[0,i,j] < 1.0
					u_plot[i,j]   = u[k,i,j]
					tau_plot[i,j] = tau[k,i,j]			       
		
		# Scatter plot, c gives the colour variable and s the size.
		plt.scatter(u_plot, tau_plot, c = H_w[k,:,:], s=1, cmap='brg')

		# Colourbar settings.
		ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
		cbar  = plt.colorbar(ticks = ticks)

		cbar.set_ticklabels(['$0.0$', '$0.1$', '$0.2$', '$0.3$', '$0.4$'])
		cbar.ax.tick_params(labelsize=17)
		cbar.ax.set_ylabel('$ H_w \ (m) $', labelpad=12, fontsize=17)

		# x ticks.
		ax.set_xticks([0, 100, 200, 300, 400, 500])
		ax.set_xticklabels(['$0$', '$100$', '$200$', \
							'$300$', '$400$', '$500$'], fontsize=17)

		# y ticks.
		ax.set_yticks([0, 50, 100, 150, 200])
		ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], fontsize=17)	

		# Axis labels.
		ax.set_xlabel(r'$ u_{b} \ (m/yr)$',fontsize=17)
		ax.set_ylabel(r'$\tau_{b} \ (kPa)$',fontsize=17)

		# Axis limits.
		ax.set_xlim(0.0, 500)
		ax.set_ylim(0.0, 200)

		#ax.set_title(title_name[ens_index],fontsize=14)

		# Axis ticks.
		ax.tick_params(axis='both', which='major', length=4, color='black')

		# Grid.
		ax.grid(which='both')
		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		##### Frame name ########
		# Start labelling on n_0.
		k_lab = k + n_0

		# Frame name to save figure.
		if k_lab < 10:
			frame = '00'+str(k_lab)
		elif k_lab > 9 and k_lab < 100:
			frame = '0'+str(k_lab)
		else:
			frame = str(k_lab)

		# Tight plot format.
		plt.tight_layout()

		# Save figures if desired.
		if save_figures == True:
			plt.savefig(path_save+ensemble_name+ \
						'_scatter_u_xy_b_tau_b_col.Hw_'+str(frame)+'.png', bbox_inches='tight')
		
		# Display and close figure.
		plt.show()
		plt.close(fig)




#########################################################################
#########################################################################

# Tau_b vs u_b colour by scaling factor lambda.

if tau_u_colour_lmbd == True:

	# Load topography for scaling factor lamdba.
	path_data  = '/home/dmoren07/yelmo-model_v1.751/yelmox/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-RTOPO-2.0.1.nc'
	nc_YELMO2D = os.path.join(get_datadir(), path_data)
	YELMO2D    = Dataset(nc_YELMO2D, mode='r')
	z_bed      = YELMO2D.variables['z_bed'][:]

	# Define lambda from z_bed.
	z_0      = - 100.0
	lmbd_max = 1.0
	lmbd     = np.empty((s[1], s[2]))

	"""
	####################################################################################
	# PLOT LAMBDA FUNCTION TO TEST.
	z_s = np.linspace(-500, 100, 100)
	lmbd_s = np.exp( - z_s / z_0 )

	fig = plt.figure(dpi=400)
	ax = fig.add_subplot(111)

	plt.rcParams['text.usetex'] = True
		       
	# Scatter plot, c gives the colour variable and s the size.
	ax.plot(z_s, lmbd_s, 'blue')

	# Tight plot format.
	plt.tight_layout()
		
	# Display and close figure.
	plt.show()
	plt.close(fig)
	####################################################################################
	"""

	# FIGURE.
	# Plot a figure per frame.
	for k in range(l_t):

		fig = plt.figure(dpi=400)
		ax = fig.add_subplot(111)

		plt.rcParams['text.usetex'] = True

		# Allocate plot variables.
		u_plot      = np.empty((s[1], s[2]))
		tau_plot    = np.empty((s[1], s[2]))
		u_plot[:]   = np.nan
		tau_plot[:] = np.nan

		# Loop over all points.
		for i in range(s[1]):
			for j in range(s[2]): 

				# Berock scaling from lambda definition.
				lmbd[i,j] = min(1.0, np.exp( - z_bed[i,j] / z_0 ))
				
				# Ensure zero value for plot.
				if lmbd[i,j] < 1.0e-4:
					lmbd[i,j] = 0.0
				
				# Avoid values below a certain threshold.
				if u[k,i,j] > 1.5 and H[k,i,j] > 1.0:   # if u[0,i,j] < 1.5 or H[0,i,j] < 1.0
					u_plot[i,j]   = u[k,i,j]
					tau_plot[i,j] = tau[k,i,j]			       
		
		# Scatter plot, c gives the colour variable and s the size.
		plt.scatter(u_plot, tau_plot, c = lmbd, s=1, cmap='brg')

		# Colourbar settings.
		ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
		cbar  = plt.colorbar(ticks = ticks)

		cbar.set_ticklabels(['$0.0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'])
		cbar.ax.tick_params(labelsize=17)
		cbar.ax.set_ylabel('$ \lambda (z_b) $', labelpad=12, fontsize=17)

		# x ticks.
		ax.set_xticks([0, 100, 200, 300, 400, 500])
		ax.set_xticklabels(['$0$', '$100$', '$200$', \
							'$300$', '$400$', '$500$'], fontsize=17)

		# y ticks.
		ax.set_yticks([0, 50, 100, 150, 200])
		ax.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], fontsize=17)	

		# Axis labels.
		ax.set_xlabel(r'$ u_{b} \ (m/yr)$',fontsize=17)
		ax.set_ylabel(r'$\tau_{b} \ (kPa)$',fontsize=17)

		# Axis limits.
		ax.set_xlim(0.0, 500)
		ax.set_ylim(0.0, 200)

		#ax.set_title(title_name[ens_index],fontsize=14)

		# Axis ticks.
		ax.tick_params(axis='both', which='major', length=4, color='black')

		# Grid.
		ax.grid(which='both')
		ax.grid(which='minor', alpha=0.2)
		ax.grid(which='major', alpha=0.5)

		##### Frame name ########
		# Start labelling on n_0.
		k_lab = k + n_0

		# Frame name to save figure.
		if k_lab < 10:
			frame = '00'+str(k_lab)
		elif k_lab > 9 and k_lab < 100:
			frame = '0'+str(k_lab)
		else:
			frame = str(k_lab)

		# Tight plot format.
		plt.tight_layout()

		# Save figures if desired.
		if save_figures == True:
			plt.savefig(path_save+ensemble_name+ \
						'_scatter_u_xy_b_tau_b_col.lmbd_'+str(frame)+'.png', bbox_inches='tight')
		
		# Display and close figure.
		plt.show()
		plt.close(fig)


