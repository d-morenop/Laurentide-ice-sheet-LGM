#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  24 10:01:40 2020

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

import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import box
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import floating_axes


################################################################################
################################################################################

# This script makes a 2D plot of any desired variable for each given
# frame of a range defined as [n_1, n_1].

########################### #####################################################
################################################################################


# OPTIONS
# Plot hudson region --> plots a black box and Hudson ice stream.
plot_hudson_region = False

# Resolution: 16, 32 km.
res = 16

# Paths to read output and save figures.
path         = '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/N.eff/'
path_fig     = '/home/dmoren07/figures/v1.751/paper.a/LIS-16KM/overburden/'
path_margold = '/home/dmoren07/ice_data/Margold-et-al_2014/'


if res == 32:

	# BC and surface elevation paths.
	path_0        = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-32KM_TOPO-RTOPO-2.0.1.nc'
	path_ice_data = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-32KM_TOPO-ICE-6G_C.nc'
	
	# Hudson subdomain vertices.
	i1 = 80
	i2 = 135 
	j1 = 61 
	j2 = 101
    
	# Time series point reference.
	i0 = 206
	j0 = 178

elif res == 16:

	# BC and surface elevation paths.
	path_0        = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-RTOPO-2.0.1.nc'
	path_ice_data = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-ICE-6G_C.nc'
    
	# Hudson subdomain vertices.
	i1 = 155  # 173
	i2 = 268  
	j1 = 127 
	j2 = 190 
    
	# Time series point reference.
	i0 = 206
	j0 = 178

ensemble = []


# List subdirectories in ensemble and order.
ensemble = os.listdir(path)
ensemble.sort()

# Now we select the ensemble (it counts batch folder as well). 
# One 2D plot per output frame will be saved for the entire integration time.
index         = 3
ensemble_name = ensemble[index]

print('')
print('Ensemble = ', ensemble_name)
print('')

# Define path to save figures from ensebmle name.
path_save = path_fig+ensemble_name+'/'

# Make output path if it does not exist.
isdir = os.path.isdir(path_save)
if isdir == False:
	os.makedirs(path_save)


# We load the initial surface elevation for visualization (land-ocean).
nc_YELMO2D  = os.path.join(get_datadir(), path_0)
YELMO2D     = Dataset(nc_YELMO2D, mode='r')
srf_0       = YELMO2D.variables['z_srf'][:]

# Time frames choice.
n_1     = 49 # 400
n_2     = 50 # 500
out_2D  = 0.4                              # period of writing 2D data (kyr).
year    = 0.01 * np.arange(n_1, n_2, 1)
n_frame = np.arange(n_1, n_2, 1)

# Particular point on the 2D plot.
x = np.array([i1, i2, i0])
y = np.array([j1, j2, j0])

# Read yelmo2D file.
path_now    = path+ensemble_name+'/'
nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
YELMO2D     = Dataset(nc_YELMO2D, mode='r')

# Get matrix dimensions.
xc = YELMO2D.variables['xc'][:]
yc = YELMO2D.variables['yc'][:]
tc = YELMO2D.variables['time'][:]
s  = np.array([len(yc),len(xc)])

# Create streaming mask.
strm_mask = np.zeros([s[0],s[1]])

l_t = len(tc)
#n_frame = np.arange(l_t-1, l_t, 1)
#n_frame = np.arange(n_1, n_2, 1)

# Appropriate coordinates to plot in Python.
x_plot = s[0] - y
y_plot = x

# Rectangular region of interest.
l_1 = np.arange(y_plot[0],y_plot[1],1)
l_2 = np.arange(x_plot[1],x_plot[0],1)
s1  = len(l_1)
s2  = len(l_2)

#####################################################################################
#####################################################################################

# File names.
file_flowlines       = "IS_flowline.shp"
file_polygons        = "IS_polygons.shp"
file_polygons_polar  = "IS_polygons_polar-stereographic.shp"

# Load Margold et al. (2014) data and transform coordinate system.
dataframe_flowlines = gpd.read_file(path_margold+file_flowlines)
dataframe_polygons  = gpd.read_file(path_margold+file_polygons)

# Transfor to yelmo LIS coordindates:
# Define the input and output coordinate systems
input_crs  = pyproj.CRS('EPSG:3978')
output_crs = pyproj.CRS('+proj=stere +lat_0=70 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')



# Define the custom polar stereographic CRS
custom_crs = pyproj.CRS.from_dict({
    'proj': 'stere',
    'lat_0': 70,
    'lat_ts': 70,
    'lon_0': -45,
    'k_0': 1,
    'x_0': 0,
    'y_0': 0,
    'a': 6378137,
    'b': 6356752.314245,
    'units': 'm',
    'no_defs': True,
    'alpha': 20
})

# Create a bounding box based on the xc and yc values
bbox = box(-4900, -5400, -36, 1576)




# Load the input data and reproject it to the output coordinate system
reprojected_dataframe = dataframe_polygons.to_crs(custom_crs)

# Save the reprojected data to a new file.
reprojected_dataframe.to_file(path_margold+file_polygons_polar)

# Read saved file.
dataframe_polygons_polar = gpd.read_file(path_margold+file_polygons_polar)

# Crop the reprojected dataframe to the bounding box
dataframe_cropped = dataframe_polygons_polar.cx[bbox.bounds[0]:bbox.bounds[2], bbox.bounds[1]:bbox.bounds[3]]




print(dataframe_polygons.total_bounds)
print(reprojected_dataframe.total_bounds)

print('CSR projection = ')
print(dataframe_polygons.crs)

print('CSR projection = ')
print(reprojected_dataframe.crs)


# Check the projection of the dataframe
print('Info polygons = ')
dataframe_polygons.info()
print('CSR projection = ')
print(dataframe_polygons.crs)

print('Info polygons_polar = ')
dataframe_polygons_polar.info()
print('CSR projection = ')
print(dataframe_polygons_polar.crs)


#####################################################################################
#####################################################################################


####################################################
################# Variable names ###################

# Available variables to be plotted.
varn = ['H_ice', 'visc_eff_int', 'f_pmp', 'H_w', \
		'uxy_bar', 'beta', 'Q_b', 'T_prime', 'taub']

# Variable options:
"""

'H_ice'        ---> 0
'visc_eff_int' ---> 1
'f_pmp'        ---> 2
'H_w'          ---> 3
'uxy_bar'      ---> 4
'beta'         ---> 5
'Q_b'          ---> 6 
'T_prime'      ---> 7
'taub'         ---> 8

"""

# Variables to be plotted. [0, 4, 7, 8]
plot_var = np.array([4])

# We plot all frames between n_1 and n_2 for each variable.
for i in plot_var:
	
	var_index = i
	var_name  = varn[i]
	print('')
	print('Variable = ', var_name)
	print('')
		
	l = len(ensemble)
	t = np.empty(l) # vector with values of temporal dimensions

	max_array = np.empty(l) # vector with highest variable value
	min_array = np.empty(l)
		
	# We load the initial surface elevation for visualization (land-ocean).
	nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
	YELMO2D     = Dataset(nc_YELMO2D, mode='r')
	t_yelmo     = YELMO2D.variables['time'][:]
	
	# We plot the last frame of the simulation.
	n = len(t_yelmo) - 1
		
	# Extent comparison with ICE-6G_C.
	#path_ice_data = '/home/dmoren07/ice_data/Laurentide/LIS-32KM/LIS-32KM_TOPO-ICE-6G_C.nc'
	varn_ice_6G = ['sftgif']		
	nc_YELMO2D  = os.path.join(get_datadir(), path_ice_data)
	YELMO2D     = Dataset(nc_YELMO2D, mode='r')
	ice_6G      = YELMO2D.variables[varn_ice_6G[0]][0,:,:]	
		
	# T_ice is 4D. z_level must be chosen. Currently plotting nz = 0, 2, 4, 9.
	z_level = 0
	print('')
	print('z_level =', z_level)
	print('')
    
    ####################################################
    ################# Max/min boundaries ###############
    
	for i in range(l):

		# Read nc file.
		#nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
		nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo_LIS_newproj.nc')
		YELMO2D     = Dataset(nc_YELMO2D, mode='r')
    
		# Temperature field has an additional dimension.
		if var_index == 7:
			var_plot = YELMO2D.variables['T_prime'][n,z_level,:,:]
		else:
			var_plot = YELMO2D.variables[var_name][n,:,:]
		
		# Variable dimensions.
		s = np.shape(var_plot)

		# Log scale just for i == 6.
		#if var_index > 3 and var_index != 7 and var_index != 4 and var_index != 8 and var_index != 5:
		if i == 6:
			for j in range(s[0]):
				for k in range(s[1]):
					if var_plot[j,k]==0.0:
						var_plot[j,k] = np.nan
		                
			var_plot = np.log10(var_plot) 
		
		# We set min/max values to plot.
		max_array[i] = np.nanmax(var_plot)
		min_array[i] = np.nanmin(var_plot)

	# Variable highest value.
	max_value = np.nanmax(max_array)
	min_value = np.nanmin(min_array)

	# Number of points in scales.
	N     = 100*2
	N_int = np.int(N)
	N_1   = np.int(0.4*N)
	N_2   = np.int(0.2*N)
	
	if var_index == 0:
		
		# Manually set min/max limits.
		lim_min = 0
		lim_max = 175

# 		levels_s = np.concatenate([[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]])
# 		colors_s =['#3B3B3B','#595959','#636363','#6E6E6E','#7D7D7D',\
# 				   '#858585','#999999','#B0B0B0','#C9C9C9','#E3E3E3','#EDEDED']
		
		# Plot colourmaps.
		levels_s = np.concatenate([[0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25]])
		colors_s =['#3B3B3B','#595959', '#636363','#7D7D7D',\
				   '#999999','#C9C9C9','#EDEDED']
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))
		
		# Load Hudson Ice Stream section from file.
		path_data = '/home/dmoren07/anaconda3/python_data/section/'
		file_name = 'cbz0.-50.cfstrm.1.0.npz' #'btmthd.1.cbz0.-050_32km_def.npz' #'cbz0.-50.cfstrm.1.0.npz'
		file      = path_data+file_name
		w         = np.load(file)
		j_tot     = w['j_tot']
		k_tot     = w['k_tot']
		l_tot     = len(j_tot)

		# Change of coordinate to plot in python.
		x_tot = np.double(s[0] - j_tot)
		y_tot = np.double(k_tot)
		
		# Avoid last points.
		x_tot[l_tot-5:l_tot] = np.nan
		y_tot[l_tot-5:l_tot] = np.nan
	
	elif var_index == 2: # f_pmp
	
		lim_min   = 0.0
		lim_med_1 = 0.4*max_value
		lim_med_2 = 0.6*max_value
		lim_max   = max_value + 1e-3
		colors1   = plt.cm.seismic(np.linspace(0.1, 0.5, N_int)) #plt.cm.plasma(np.linspace(0.02, 1.0, N-1))      
		levels    = np.concatenate([np.linspace(lim_min,lim_max,N_int+1)])
		colors    = np.vstack((colors1))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels, colors)
		cmap_v.set_over(plt.cm.RdYlBu(0))
		
	elif var_index == 3: # H_w
	
		lim_min   = 0.0
		lim_med_1 = 0.4*max_value
		lim_med_2 = 0.6*max_value
		lim_max   = max_value + 1e-3
		colors1   = plt.cm.Greens(np.linspace(0.0, 0.75, N_int)) #plt.cm.plasma(np.linspace(0.02, 1.0, N-1))      
		levels    = np.concatenate([np.linspace(lim_min,lim_max,N_int+1)])
		colors    = np.vstack((colors1))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels, colors)
		cmap_v.set_over(plt.cm.RdYlBu(0))
		
	elif var_index == 4:

		lim_min = 0
		lim_max = 2000
		levels_s = np.concatenate([[0, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]])
		colors_s =['#ffffff','#d9c8e0','#bcb3d4','#9087ba','#61409b',\
				   '#00b1ff','#ffef37','#e0d018','#ffa200','#ff0000']
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))
		
		# Streaming mask settings.
		strm_lim_min = 0
		strm_lim_max = 1.1
		strm_levels_s = np.concatenate([[0.0, 0.25, 0.5, 0.75, 1.1]])
		#strm_colors_s =['#d9d9d9','#d9d9d9','#ffa200','#ffa200'] #a6a6a6
		#strm_colors_s =['#ffffff', '#ffffff','#008000', '#008000'] #a6a6a6
		strm_colors_s =['#ffffff', '#ffffff','#008000', '#008000'] #a6a6a6
		strm_cmap, strm_norm = mcolors.from_levels_and_colors(strm_levels_s, strm_colors_s)
		strm_cmap.set_over(plt.cm.jet(240))
		strm_cmap_v, strm_norm_v = mcolors.from_levels_and_colors(strm_levels_s, strm_colors_s)
		strm_cmap_v.set_over(plt.cm.jet(240))
		

	elif var_index == 5:
	
		lim_min = 1.0e2
		lim_max = 1.0e8
		levels_s = np.concatenate([[1.0e2,1.0e3,1.0e4,1.0e5,1.0e6,5.0e6,1.0e7,5.0e7,5.0e8]])
		colors_s =['#ffc4fe','#c36ef5','#61409b',\
				   '#9087ba','#2800ed','#0086ed','#00beed','#00ede9']
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))
		
	elif var_index == 7:
		
		lim_min = -15
		lim_max = 0
		levels_s = np.concatenate([[-15,-12.5,-10,-7.5,-5,-4,-3,-2,-1,-0.1,0.0,0.1]])
		colors =['#e88000','#ffb300','#fbff00','#ccff00','#84ff00','#00c227',\
				   '#00ffee','#59d6ff','#0091ff','#0055ff','#aa00ff']
		colors_s = colors[::-1]
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))
		
	elif var_index == 8:
		
		lim_min = 0
		lim_max = 175
		levels_s = np.concatenate([[0,10,20,30,35,40,50,60,75,100,125]])
		colors_s =['#ffffff','#b8f7b7','#84e082','#55cf53','#1bad18',\
				   '#1b8ce3','#ffef37','#e0d018','#ffa200','#ff0000']
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))
		
	else:
	
		lim_min = 0
		lim_max = 1.0e8
		levels_s = np.concatenate([[0,1.0e1,5.0e1,1.0e2,1.0e3,1.0e4,1.0e5,1.0e6,1.0e7,5.0e7,1.0e8]])
		colors_s =['#ffffff','#d9c8e0','#bcb3d4','#9087ba','#61409b',\
				   '#7300ed','#2800ed','#0086ed','#00beed','#00ede9']
		cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap.set_over(plt.cm.jet(240))
		cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
		cmap_v.set_over(plt.cm.jet(240))



	######################################################################
	########### One plot for each frame within [n_1, n_2] #################

	for i in n_frame:

		# Read yelmo output file.
		#nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
		nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo_LIS_newproj.nc')
		YELMO2D     = Dataset(nc_YELMO2D, mode='r')
		H_ice       = YELMO2D.variables['H_ice'][i,:,:]
		f_grnd      = YELMO2D.variables['f_grnd'][i,:,:]
		
		# Define ration eps = sliding / creeping velocities.
		eps    = np.empty([s[0], s[1]])
		eps[:] = np.nan
	
		# Ice temperature.
		if var_index == 7:
			var_plot = YELMO2D.variables['T_prime'][i,z_level,:,:]	
		
		# Grounded fraction.
		elif var_index == 2:
			var_plot = YELMO2D.variables[var_name][i,:,:]
			f_grnd   = YELMO2D.variables['f_grnd'][i,:,:]
		
		# Velocity and streaming mask.
		elif var_index == 4:
			strm_mask[:] = np.nan
			var_plot = YELMO2D.variables[var_name][i,:,:]
			u_b      = YELMO2D.variables['uxy_b'][i,:,:]
			u_bar    = YELMO2D.variables['uxy_bar'][i,:,:]
			
			# Deformation (creeping) velocity definition u_def.
			u_def = u_bar - u_b

			# Mask settings.
			# Minimum factor to consider streaming.
			eps_thres = 2.0

			# Minimum sliding velocity to avoid noise.
			u_b_min = 50.0
			
			# Create streaming mask. Binary, 1 = streaming.
			for j in range(s[0]):
				for k in range(s[1]):
					
					# Avoid points with no ice.
					if H_ice[j,k] == 0:
						strm_mask[j,k] = np.nan
					
					# Mask out noisy north region.
					elif j > 270 and H_ice[j,k] < 400.0: 
						var_plot[j,k] = np.nan
						strm_mask[j,k] = np.nan
					
					# Eps defined for non-zero deformation velocity.
					elif u_def[j,k] != 0.0:
						
						# Ratio definiton.
						eps[j,k] = u_b[j,k] / u_def[j,k]

						# Ratio above threshold and minum u_b value.
						if eps[j,k] > eps_thres and u_b[j,k] > u_b_min:
							strm_mask[j,k] = 1
						else:
							strm_mask[j,k] = np.nan
					
					# Floating points with non-zero ice are considered to be streaming.
					elif H_ice[j,k] != 0 and f_grnd[j,k] == 0:
 						strm_mask[j,k] = 1
					else:
						strm_mask[j,k] = np.nan

		# Rest of variables.
		else:
			var_plot = YELMO2D.variables[var_name][i,:,:]
	
		# Print variable index.
		print('i = '+str(i))
	
		
		# Additional calculation for each variable.
		# Log scale.
		if i == 6:
		#if var_index > 3 and var_index != 7 and var_index != 4 and var_index != 8 and var_index != 5:
			for j in range(s[0]):
				for k in range(s[1]):
					if var_plot[j,k] == 0.0 or f_grnd[j,k] == 0:
						var_plot[j,k] = np.nan
	
			# We represent in log scale.
			var_plot = np.log10(var_plot)
		
		elif var_index == 7:
			for j in range(s[0]):
				for k in range(s[1]):
					if f_grnd[j,k] == 0 or H_ice[j,k] == 0.0:
						var_plot[j,k] = np.nan
		
		elif var_index == 0 or var_index == 8:
			var_plot = 1.0e-3 * var_plot
			for j in range(s[0]):
				for k in range(s[1]):
					if f_grnd[j,k] == 0 or H_ice[j,k] == 0.0:
						var_plot[j,k] = np.nan


		# Grounding line is loaded for f_pmp plots.
		if var_index == 2 or var_index == 3:
			nc_YELMO2D = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
			YELMO2D    = Dataset(nc_YELMO2D, mode='r')
			f_grnd     = YELMO2D.variables['f_grnd'][n,:,:]	
			if var_index == 3:
				for j in range(s[0]):
					for k in range(s[1]):
						if f_grnd[j,k] == 0:
							var_plot[j,k] = np.nan
			

		
		###################################################################################
		###################################################################################
		#                                 FIGURES
		
		# Create figure and axis as a subplot.
		fig = plt.figure(dpi=400)
		ax  = fig.add_subplot(111)
			
		# Use latex as text interpreter.
		plt.rcParams['text.usetex'] = True
		
		# Plot Hudson subdomain if chosen.
		if plot_hudson_region == True:
			
			# Plot Hudson Strait ice stream.
			ax.plot(x_tot, y_tot, linestyle='-', color='blue', marker= 'None', \
					markerfacecolor='blue', linewidth=3.5, alpha=1.0)

			# Plot point taken for time series.
			ax.plot(x_plot[2], y_plot[2], linestyle='none', color='black', marker= 's', \
					markerfacecolor='white', markersize=5.5, alpha=1.0)

			# Plot black box encompassing Hudson subdomain.
			for j in range(s1):
				ax.plot(x_plot[0], l_1[j], linestyle='none', color='black', marker= 'o', \
						markerfacecolor='darkgrey', markersize=1, alpha=1.0)
				ax.plot(x_plot[1], l_1[j], linestyle='none', color='black', marker= 'o', \
						markerfacecolor='darkgrey', markersize=1, alpha=1.0)
			for j in range(s2):
				ax.plot(l_2[j], y_plot[0], linestyle='none', color='black', marker= 'o', \
						markerfacecolor='darkgrey', markersize=1, alpha=1.0)
				ax.plot(l_2[j], y_plot[1], linestyle='none', color='black', marker= 'o', \
						markerfacecolor='darkgrey', markersize=1, alpha=1.0)			

		
		# Paper.
		#extent = [1.4*xmin, 80.0*xmax, 1.3*ymin, 1.0*ymax] # horizontal
		extent = [1.4*xmin, 50.0*xmax, 1.05*ymin, 1.0*ymax] 
		
		# Plot 2D variables.	
		im = ax.imshow(np.flip(np.rot90(var_plot,3), axis=0), cmap=cmap_v, norm=norm_v, extent=extent)
		
		# Colour settings.
		colors_s       = plt.cm.BrBG(np.linspace(0.3, 0.0, N, endpoint=True))
		levels_s       = np.concatenate([np.linspace(lim_min, lim_max, N+1)])
		cmap_s, norm_s = mcolors.from_levels_and_colors(levels_s, colors_s)
		
		# Contour plots.
		contour       = np.round(np.arange(0.5, 5.6, 1.0), decimals=1)
		H_ice_contour = np.round(1.0e-3*np.rot90(H_ice, 3), decimals=1)
		CS            = ax.contour(H_ice_contour, contour, linewidths=0.75,\
							  	   linestyles='--', colors='black', norm=norm_s, extent=extent)
		
		# Label in contour lines.
		ax.clabel(CS, inline=1, fmt='%1.1f', fontsize=10)
		
		# Plot surface elevation contour field.
		ax.contour(np.rot90(srf_0,3), np.linspace(0,0,1), linewidths=1.0,\
				   linestyles='-', colors='black', norm=norm_s, extent=extent)	
		
		# Extent comparison with ICE-6G_C. Variable given in %, we choose 50% as a threshold.
		ax.contour(np.rot90(ice_6G,3), np.linspace(50,50,1), linewidths=1.5,\
				   linestyles='-.', colors='red', norm=norm_s, extent=extent)
		
		# New axis dimensions.
		#cax = fig.add_axes([0.92, 0.13, 0.03, 0.75])
		cax = fig.add_axes([0.84, 0.13, 0.03, 0.75])

		# Colourbar.
		ticks = np.linspace(lim_min,np.rint(lim_max),10)
		cb    = fig.colorbar(im, cax=cax, extend='neither', ticks=ticks)
		#cb    = fig.colorbar(im, extend='neither', ticks=ticks)
			
		# varn = ['H_ice','visc_eff','f_pmp','H_w','uxy_bar','beta','Q_b']
		if var_index == 0:                                                
			cb.set_ticks([0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25]) #H_ice
			cb.set_ticklabels(['$0$', '$0.75$', '$1.5$', '$2.25$', '$3.0$', '$3.75$', \
							   '$4.5$', '$5.25$'])
			cb.ax.tick_params(labelsize=17)
			cb.set_label('$H_{\mathrm{ice}}$ $(km)$', rotation=270, labelpad=30, fontsize=22)
			    
		elif var_index == 1:
			cb.set_ticks([0, 0.1e13, 0.2e13, 0.3e13, 0.4e13, 0.5e13, 0.6e13, 0.7e13, \
						  0.8e13, 0.9e13, 1.0e13, 1.1e13]) #effective viscosity
			cb.ax.tick_params(labelsize=17)
			cb.set_label('$\eta$ $(Pa \cdot yr \cdot m)$', rotation=270,labelpad=20,fontsize=22)
			    
		elif var_index == 2:
			cb.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) #f_pmp
			cb.ax.tick_params(labelsize=14)
			cb.set_label('$f_{pmp}$', rotation=270,labelpad=15,fontsize=22)
			# f_grnd comparison. Discrete variable {0,1}. 1 = grounded.
			ax.contour(np.rot90(f_grnd,3),np.linspace(0.9,1.1,1),linewidths=2.5,\
						  linestyles=':',colors='green',norm=norm_s)
			    
		elif var_index == 3:
			#cb.set_ticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]) # H_w
			cb.set_ticks([0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40])
			cb.set_ticklabels(['$0$', '$0.05$','$0.10$','$0.15$', '$0.20$',\
								  '$0.25$', '$0.30$', '$0.35$', '$0.40$'])
			cb.ax.tick_params(labelsize=17)
			cb.set_label('$H_{w}$ $(m)$', rotation=270,labelpad=20,fontsize=22)
			    
		elif var_index == 4:
			cb.set_ticks([0,2,5,10,20,50,100,200,500,1000,2000])
			cb.set_ticklabels(['$0$', '$2$', '$5$', '$10$', '$20$', '$50$',\
								  '$100$', '$200$', '$500$', '$1000$', '$2000$'])
			cb.ax.tick_params(labelsize=17)
			cb.set_label('$\overline{u}$ $(m/yr)$', rotation=270,labelpad=24,fontsize=22)

		elif var_index == 5:
			cb.set_ticks([1.0e2,1.0e3,1.0e4,1.0e5,1.0e6,5.0e6,1.0e7,5.0e7,5.0e8]) #beta
			cb.set_ticklabels(['$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$',\
								  '$10^{6}$','$5 \cdot 10^{6}$','$1 \cdot 10^{7}$',\
									  '$5 \cdot 10^{7}$','$5 \cdot 10^{8}$'])
			cb.ax.tick_params(labelsize=17)
			cb.set_label(r'$ \beta \ (Pa \cdot yr / m)$', rotation=270,labelpad=24,fontsize=22)
	
		elif var_index == 6:
			cb.set_ticks([0,1,2,3,4,5,6,7]) #Q_b
			cb.ax.tick_params(labelsize=17)
			cb.set_ticklabels(['$1$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$',\
								  '$10^{5}$','$10^{6}$','$10^{7}$'])
			cb.set_label('$Q_{b}$ $(J \cdot m^{2} / yr)$', rotation=270,labelpad=20,fontsize=22) 

		elif var_index == 7:
			#title = str(title_name[index])+' - $z_{n}= $'+str(z_level)                                          
			cb.set_ticks([-15,-12.5,-10,-7.5,-5,-4,-3,-2,-1,-0.1,0.0,0.1])
			cb.ax.tick_params(labelsize=17)
			cb.set_label('$T_{h}$ $(^{\circ} C)$', rotation=270,labelpad=20,fontsize=22)
		
		elif var_index == 8 :                                       
			cb.set_ticks([0,10,20,30,35,40,50,60,75,100,125])
			cb.set_ticklabels(['$0$', '$10$', '$20$', '$30$', '$35$', '$40$',\
								  '$50$', '$60$','$75$', '$100$', '$125$'])
			cb.ax.tick_params(labelsize=17)
			cb.set_label(r'$\tau_{b}$ $(kPa)$', rotation=270,labelpad=20,fontsize=22)
	

		##### Frame name ########
		if i < 10:
			frame = '00'+str(i)
		elif i > 9 and i < 100:
			frame = '0'+str(i)
		else:
			frame = str(i)
			
		#ax.invert_yaxis()
		ax.set_yticklabels([])
		ax.set_xticklabels([])

		ax.tick_params(axis='both', which='major', length=0, colors='black')
		
		#ax.set_title(r'$n = $'+str(i-n_1)+r'$, \ t = $'+str(np.round(1.0e-3 * t_yelmo[i],1))+r'$ \ kyr$', fontsize=18)
		#plt.tight_layout()

		if var_index == 7:
			plt.savefig(path_save+var_name+'_zn.'+str(z_level)+\
            			   '_'+ensemble_name+'_'+str(frame)+'.png', bbox_inches='tight')
		else:
			plt.savefig(path_save+var_name+'_'+ensemble_name+\
            			   '_'+str(frame)+'.png', bbox_inches='tight')
	
		plt.show()
		plt.close(fig)
	
		# We now plot the stream mask.
		if var_index == 4:

			# Create figure and axis as a subplot.
			fig = plt.figure(dpi=400)
			ax  = fig.add_subplot(111)
			
			# Use latex as text interpreter.
			plt.rcParams['text.usetex'] = True
			
			# Manually set the extent of the image to match the original dimensions of the image
			xmin, ymin, xmax, ymax = dataframe_polygons_polar.total_bounds
			extent = [xmin, xmax, ymin, ymax]
			
			# Plot Yelmo field.
			#im = ax.imshow(np.rot90(strm_mask, 3), cmap=strm_cmap_v, norm=strm_norm_v)

			# Manually match extents.
			#extent = [1.2*xmin, xmax, 1.05*ymin, 1.55*ymax] # vertical
			# extent = [1.4*xmin, 50.0*xmax, 1.0*ymin, 0.95*ymax] # horizontal
			
			extent = [1.4*xmin, 50.0*xmax, 1.05*ymin, 1.0*ymax] # horizontal
			
			# Paper.
			#extent = [1.4*xmin, 80.0*xmax, 1.3*ymin, 1.0*ymax] # horizontal

			#extent = [1.4*xmin, 50.0*xmax, 1.0*ymin, 1.0*ymax] # horizontal

			im = ax.imshow(np.flip(np.rot90(strm_mask, 3),axis=0), cmap=strm_cmap_v, norm=strm_norm_v, \
									extent=extent, zorder=4, alpha=0.9)
			
	

			# Include Margold et al. (2014)
			#dataframe_polygons_polar.plot(column="geometry", ax=ax)
			# create a new dataframe that only contains the entries you want to plot
			data_ID  = dataframe_polygons_polar['Stable_ID']
			l_df     = len(data_ID)
			ID_avoid = ['179', '180', '165', '33', '26', '164', '166', '174']

			# To delete the deglatiation ice streams we plot them in white grey
			# so that it can't be seen and the size is not altered.
			ID_white = ['179', '180', '165', '33', '26', '164', '166', '174',\
						'6', '143', '144', '175', '145', '176', '147', '148', '177', \
						'178', '157', '156', '151', '150', '156', '155', '152', '153', \
						'154', '160', '158', '161', '14', '15', '153', '162', '159', \
						'163']
			
			# Boolean array of the same dimension as a Python list with false 
			# entry values corresponding to the positions where the 
			# list has any of a certain set of target entries, you can use the in 
			# operator and another list comprehension.
			avoid_array = [entry not in ID_avoid for entry in data_ID]
			white_array = [entry not in ID_white for entry in data_ID]


			new_df   = dataframe_polygons_polar.loc[avoid_array]
			white_df = dataframe_polygons_polar.loc[white_array]
			
			#dataframe_polygons.plot(column="geometry", ax=ax, \
			#				linewidth=0.4, alpha=1.0, color='None', edgecolor='black', \
			#					hatch='.....', zorder=3)

			# Define colours.
			colors = ['#8c8c8c' if val else '#e1e1e1' for val in white_array]

			polygons_rotated = dataframe_polygons_polar.rotate(90, origin=new_df.unary_union.centroid)
			new_df_rotated   = new_df.rotate(90, origin=new_df.unary_union.centroid)
			white_rotated    = white_df.rotate(90, origin=white_df.unary_union.centroid)
			
			#new_df_rotated.plot(ax=ax, linewidth=0.4, alpha=1.0, color='None', \
			#						edgecolor='grey', hatch='.....', zorder=4)
			
			#white_rotated.plot(ax=ax, linewidth=0.4, alpha=1.0, color='None', \
			#						edgecolor='black', hatch='.....', zorder=3)


			polygons_rotated.plot(ax=ax, linewidth=0.5, alpha=1.0, \
										color=colors, edgecolor=colors, \
												hatch='....', zorder=3)

			for patch in ax.patches:
				patch.set_hatch_color(patch.get_hatch())

			# Colour settings.
			colors_s       = plt.cm.BrBG(np.linspace(0.3, 0.0, N, endpoint=True))
			levels_s       = np.concatenate([np.linspace(strm_lim_min, strm_lim_max, N+1)])
			cmap_s, norm_s = mcolors.from_levels_and_colors(levels_s, colors_s)
			

			# Contour fields.
			ax.contour(np.rot90(ice_6G, 3), np.linspace(50, 50, 1), linewidths=1.25,\
						 linestyles='-.', colors='red', norm=norm_s, extent=extent, zorder=2)
			

			ax.contour(np.rot90(srf_0, 3), np.linspace(0, 0, 1), linewidths=0.75,\
						 linestyles='-', colors='black', norm=norm_s, extent=extent, zorder=1)	


			
			# Colourbar ticks.
			ticks = np.linspace(strm_lim_min, np.rint(strm_lim_max), 10)
			#cb    = fig.colorbar(im, cax=cax, extend='neither', ticks=ticks) # pad???
			cb    = fig.colorbar(im, extend='neither', ticks=ticks)
	
			# Set ticks in colourbar.
			cb.set_ticks([0.4, 0.95])
			cb.ax.tick_params(length=0)

			# Set colourbar labels.
			cb.ax.set_yticklabels([r'$u_{\mathrm{b}}/u_{\mathrm{def}} < 10$',\
							       r'$u_{\mathrm{b}}/u_{\mathrm{def}} > 10$'], \
								   rotation=270)
			# Label fontsize.
			cb.ax.tick_params(labelsize=17)
			

			##### Frame name ########
			if i < 10:
				frame = '00'+str(i)
			elif i > 9 and i < 100:
				frame = '0'+str(i)
			else:
				frame = str(i)
			
			# Axis settings and labels.
			#ax.invert_yaxis()
			ax.set_yticklabels([])
			ax.set_xticklabels([])

			ax.tick_params(axis='both', which='major', length=0, colors='black')

			# Figure title.
			#ax.set_title(r'$n = $'+str(i-n_1)+r'$, \ t = $'+str(np.round(1.0e-3 * t_yelmo[i],1))+r'$ \ kyr$', fontsize=18)
			
			# Tight format.
			#plt.tight_layout()
	
			# Save figure in path_save.
			plt.savefig(path_save+'strm_mask_'+ensemble_name+\
        				   '_'+str(frame)+'_margold.png', bbox_inches='tight')

			# Display figure.
			plt.show()

			# Close figure.
			plt.close(fig)

print('')
print('Path_save = ', path_save)














