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
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
plt.style.use("seaborn-white")



path     = '/home/dmoren07/yelmo-model_v1.751/yelmox/output/LIS-16KM/paper.a/overburden/'
path_fig = '/home/dmoren07/figures/v1.751/paper.a/LIS-16KM/overburden/'



res = 16
if res == 32:
    path_0        = '/home/dmoren07/ice_data/Laurentide/LIS-32KM/LIS-32KM_TOPO-RTOPO-2.0.1.nc'
    path_ice_data = '/home/dmoren07/ice_data/Laurentide/LIS-32KM/LIS-32KM_TOPO-ICE-6G_C.nc'
    file_name = 'btmthd.1.cbz0.-050_32km_def.npz'
    
    delta_x = 32
    i1 = 80
    i2 = 135 
    j1 = 61 
    j2 = 101
    
    i0 = 206
    j0 = 178
    
    k_0   = 140 #127 # i_0 = 246
    j_0   = 70  # 70
elif res == 16:
    path_0        = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-RTOPO-2.0.1.nc'
    path_ice_data = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-ICE-6G_C.nc'
    file_name     = 'cbz0.-50.cfstrm.1.0.npz'
    
    delta_x = 16
    i1 = 155  # 173
    i2 = 268  
    j1 = 127 
    j2 = 190 
    
    i0 = 206
    j0 = 178
    
    k_0   = 278 # i_0 = 246
    j_0   = 143




# List all subdirectories using os.listdir
ensemble = []
ensemble = os.listdir(path)
ensemble.sort()

# Choose ensemble
index = 3
ensemble_name = ensemble[index]

print('')
print('Ensemble = ', ensemble_name)
print('')

path_save = path_fig+ensemble_name+'/'


isdir = os.path.isdir(path_save)
if isdir == False:
	os.makedirs(path_save)

print('')
print('path_save = ', path_save)
print('')

# Read ice stream location from regularized-Coulomb case.
read = 1
if read == 1:
	path_data = '/home/dmoren07/anaconda3/python_data/section/'
	#file_name = 'btmthd.1.cbz0.-050_32km_def.npz' #'cbz0.-50.cfstrm.1.0.npz'
	file      = path_data+file_name
	w         = np.load(file)
	j_read    = w['j_tot']
	k_read    = w['k_tot']
	l_read    = len(k_read) - 1

# Save curve in the map following the max value of u.
write = 0
new   = 0
path_data = '/home/dmoren07/anaconda3/python_data/section/'


# We load the initial surface elevation for visualization (land-ocean).
nc_YELMO2D  = os.path.join(get_datadir(), path_0)
YELMO2D     = Dataset(nc_YELMO2D, mode='r')
srf_0       = YELMO2D.variables['z_srf'][:]
z_bed       = YELMO2D.variables['z_bed'][:]


n_1 = 46 # 499
n_2 = 47 # 500
out_2D  = 0.4                              # period of writing 2D data (kyr).
year    = 0.01 * np.arange(n_1, n_2, 1)
n_frame = np.arange(n_1, n_2, 1)



# Define dimension from particular ensemble.
nc_YELMO2D  = os.path.join(get_datadir(), path+ensemble_name+'/yelmo2D.nc')
YELMO2D     = Dataset(nc_YELMO2D, mode='r')

xc = YELMO2D.variables['xc'][:]
yc = YELMO2D.variables['yc'][:]
s  = np.array([len(yc),len(xc)])



x_plot = s[0] - j_0
y_plot = k_0


z_0      = -100.0    # Bedreock scaling
lmbd_min = 1.0e-5

path      = [path+ensemble_name, \
             path+ensemble_name, \
             path+ensemble_name]

path_plot = np.array([0])
varn      = ['H_ice', 'uxy_bar', 'uxy_b', 'taub', 'f_grnd', 'N_eff']

plot_var = np.array([4])


for p in path_plot:
	nc_YELMO2D  = os.path.join(get_datadir(), path[p]+'/yelmo2D.nc')
	YELMO2D     = Dataset(nc_YELMO2D, mode='r')
    #print('path = ', path[p])
	#data.append(YELMO2D.variables[i][n_frame,:,:])

	######################################################################
	########### One plot for each frame within [n_1,n_2] #################

	for i in n_frame:
		
		j_tot = []
		k_tot = []
		d_tot = []
		srf   = []
		sect  = []
		u     = []
		u_b   = []
		u_SIA = []
		H     = []
		tau   = []
		N     = []
		lmbd  = []


		#print('Frame n = ', i)
		#nc_YELMO2D  = os.path.join(get_datadir(), path_now+'yelmo2D.nc')
		#YELMO2D     = Dataset(nc_YELMO2D, mode='r')
		H_ice       = YELMO2D.variables['H_ice'][i,:,:]
		f_grnd      = YELMO2D.variables['f_grnd'][i,:,:]
		uxy         = YELMO2D.variables['uxy_bar'][i,:,:]
		uxy_b       = YELMO2D.variables['uxy_b'][i,:,:]
		tau_b       = YELMO2D.variables['taub'][i,:,:]
		f_grnd      = YELMO2D.variables['f_grnd'][i,:,:]
		N_eff       = YELMO2D.variables['N_eff'][i,:,:]
		
		u_def = uxy - uxy_b
		
		c = 0
		g = 0
		
        # Calculate path if not loaded.
		if read == 0:
			for k in range(i1, i2, 1):
				u_max = np.max(uxy[j1:j2,k]) 
				
				if u_max > 20 and u_max < 1350:
					u_list        = list(uxy[:,k])
					j_max         = u_list.index(u_max)
					j_tot.append(j_max)
					k_tot.append(k)
					d             = np.sqrt( (j_max - j_0)**2 + (k - k_0)**2 )
					H.append(H_ice[j_max,k])
					u.append(u_max)
					u_b.append(uxy_b[j_max,k])
					u_SIA.append(u_def[j_max,k])
					tau.append(tau_b[j_max,k])
					N.append(N_eff[j_max,k])
					d_tot.append(d)
					srf.append(z_bed[j_max,k])
					if f_grnd[j_max,k] == 0 and g == 0:
						d_grnd = d * delta_x # PROBLEM HERE
						j_grnd = j_max
						k_grnd = k
						g = g + 1
						#exit
						
# 				elif u_max > 1350 and c < 9:
#  					c = c + 1
#  					j_m = j_max - 1
#  					j_p = j_max + 1
#  					u_max  = np.max(uxy[j_m:j_p, k])
#  					u_list = list(uxy[:,k])
#  					j_max  = u_list.index(u_max)
#  					j_tot.append(j_max)
#  					k_tot.append(k)
#  					d      = np.sqrt( (j_max - j_0)**2 + (k - k_0)**2 )
#  					d_tot.append(d)
#  					H.append(H_ice[j_max,k])
#  					u.append(u_max)
#  					u_b.append(uxy_b[j_max,k])
#  					u_SIA.append(u_def[j_max,k])
#  					tau.append(tau_b[j_max,k])
#  					N.append(N_eff[j_max,k])
#  					srf.append(z_bed[j_max,k])
#  					if f_grnd[j_max,k] == 0 and g == 0:
# 						d_grnd = d
# 						j_grnd = j_max
# 						k_grnd = k
# 						g = g + 1
		if read == 1:
			m = 0
			g = 0
			for k in range(i1, i2, 1):
				for j in range(j1, j2, 1):
					if k == k_read[m] and j == j_read[m]:
						d = np.sqrt( (j - j_0)**2 + (k - k_0)**2 )
						H.append(H_ice[j,k])
						u.append(uxy[j,k])
						u_b.append(uxy_b[j,k])
						u_SIA.append(u_def[j,k])
						tau.append(tau_b[j,k])
						N.append(N_eff[j,k])
						d_tot.append(d)
						srf.append(z_bed[j,k])
						if z_bed[j,k] >= 0.0:
							lmbd.append(1.0)
						else:
							lmbd_exp = max(np.exp(-z_bed[j,k]/z_0),lmbd_min)
							lmbd.append(lmbd_exp)
						m = min(m + 1, l_read)
						if f_grnd[j,k] == 0 and g == 0:
							d_grnd = d * delta_x
							j_grnd = j
							k_grnd = k
							g = g + 1
		
		d_tot = delta_x * np.array(d_tot)
		u     = 1.0e-3 * np.array(u)
		u_b   = 1.0e-3 * np.array(u_b)
		u_SIA = np.array(u_SIA)
		H     = 1.0e-3 * np.array(H)
		tau   = 1.0e-3 * np.array(tau)
		N     = 1.0e-6 * np.array(N)
		lmbd  = np.array(lmbd)
		k_tot = np.array(k_tot)
		j_tot = np.array(j_tot)
		srf   = 1.0e-3 * np.array(srf)
		H_srf = H + srf
		H_sh  = - H
		#H_sh  = H
		
		l        = len(d_tot)
		l_plot   = l - 1
		d_plot   = d_tot - d_grnd
		#d_plot   = d_tot - d_tot[len(d_tot)-1]
		d_plot   = - d_plot           # Invert to plot.
		d_list   = list(d_plot)
		d_zero   = d_list.index(0.0)
		
	
		
		if write == 1:
			x = [None]*12
			x[0]  = u
			x[1]  = u_b
			x[2]  = u_SIA
			x[3]  = tau
			x[4]  = N
			x[5]  = H_srf
			x[6]  = H_sh
			x[7]  = d_plot
			x[8]  = d_list
			x[9]  = srf
			x[10] = j_tot
			x[11] = k_tot
			
			file      = path_data+ensemble_name+'_32km_def.npz'
			if new == 1:
				os.makedirs(path_data)
			
			np.savez(file, u = x[0], u_b = x[1], u_SIA = x[2], tau = x[3], N = x[4],\
					 		 H_srf = x[5], H_sh = x[6], d_plot = x[7], d_list = x[8], srf = x[9],\
							  j_tot = x[10], k_tot = x[11])
		
		###################################################################################
		###################################################################################
		#                                 FIGURES
		
		fig = plt.figure(dpi=600, figsize=(5,7)) # (5,7)
		ax  = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)
		
		ax3 = ax.twinx()
		ax4 = ax2.twinx()
		
		host = host_subplot(313, axes_class=AA.Axes)
		ax6 = host.twinx()
		ax7 = host.twinx()

		# Distance between axis and parasite axis
		offset = 55 # 30, 50
		new_fixed_axis    = ax7.get_grid_helper().new_fixed_axis
		ax7.axis["right"] = new_fixed_axis(loc="right", axes=ax7,\
                                        offset=(offset, 0))
		ax6.axis["right"].toggle(all=True)
		ax7.axis["right"].toggle(all=True)


		plt.rcParams['text.usetex'] = True
		
		# Shelf plots.
		d_plt_shlf  = np.arange(0, d_plot[l_plot-1], 1)
		H_shlf_0    = np.empty(len(d_plt_shlf))
		H_shlf_0[:] = H_srf[d_zero]
		#H_srf = np.where(H_srf > 0.0, H_srf, np.nan)
		#H_shlf_0[:] = H_srf[d_zero]
		
		ax.plot(d_plt_shlf, H_shlf_0, linestyle='-', color='darkgreen', marker= 'None', \
 		     markerfacecolor='darkgreen', linewidth=2.5, markersize=3, alpha=1.0)
		ax.plot(d_plot[d_zero:l_plot], H_sh[d_zero:l_plot], linestyle='-', color='darkgreen', marker= 'None', \
 		     markerfacecolor='darkgreen', linewidth=2.5, markersize=3, alpha=1.0)
		
		# Panel 1.
		ax.plot(d_plot[0:d_zero+1], H_srf[0:d_zero+1], linestyle='-', color='darkgreen', marker= 'None', \
 		     markerfacecolor='darkgreen', linewidth=2.5, markersize=3, alpha=1.0)

		ax.fill_between(d_plot[0:d_zero+1], srf[0:d_zero+1], H_srf[0:d_zero+1],\
 						   facecolor='grey',alpha=0.4)
		ax.fill_between(d_plot[d_zero:l_plot], H_sh[d_zero:l_plot], 0,\
 						   facecolor='grey',alpha=0.4)
		
		ax.plot(d_plot, srf, linestyle='-', color='brown', marker= 'None', \
 		     markerfacecolor='darkgreen', linewidth=2.5, markersize=3, alpha=1.0)
		
		# Pannel 2.	
# 		ax2.plot(d_plot, u, linestyle='-', color='blue', marker= 'None', \
#  		     markerfacecolor='blue', linewidth=2.5, markersize=3, alpha=1.0, label=r'$\overline{u}$')
		ax2.plot(d_plot, u_b, linestyle='-', color='blue', marker= 'None', \
 		     markerfacecolor='blue', linewidth=2.5, markersize=3, alpha=1.0, label=r'$u_{b}$')
		ax4.plot(d_plot, u_SIA, linestyle=':', color='blue', marker= 'None', \
 		     markerfacecolor='blue', linewidth=2.5, markersize=3, alpha=1.0, label=r'$u_{def}$')
		
		# Pannel 3.
		host.plot(d_plot, tau, linestyle='-', color='black', marker= 'None', \
				  label= '$\tau_{b}$ $(kPa)$', markerfacecolor='black', linewidth=2.5, \
					  markersize=3, alpha=1.0)
		ax6.plot(d_plot, N, linestyle='-', color='purple', marker= 'None', \
				  label= '$N_{\mathrm{eff}}$ $(MPa)$', markerfacecolor='black', linewidth=2.5, \
					    markersize=3, alpha=1.0)
		ax7.plot(d_plot, lmbd, linestyle='-', color='darkred', marker= 'None', \
				  label= '$\lambda$',markerfacecolor='black', linewidth=2.5, \
 					    markersize=3, alpha=1.0)

		# Legend.
		p1 = ax2.plot(-10, -10, linestyle='-', marker='None', color='blue',\
 			  markerfacecolor='None', linewidth=3)
		p2 = ax2.plot(-10, -10, linestyle=':', marker='None', color='blue',linewidth=3)
		ax2.legend([(p1[0]),(p2[0]),  ],['$u_{\mathrm{b}}$', '$u_{\mathrm{def}}$'], \
						loc='upper left', ncol = 1, frameon = True, \
 							framealpha = 1.0, fontsize = 14, fancybox = True)	


		ax.set_xlim(d_plot[l_plot-1], d_plot[0])
		ax2.set_xlim(d_plot[l_plot], d_plot[0]) 
		host.set_xlim(d_plot[l_plot], d_plot[0])
		ax.set_ylim(-2.5,5.0) 
		ax2.set_ylim(-0.2,1.7)
		ax3.set_ylim(-2.5,5.0) 
		ax4.set_ylim(-10,115)
		host.set_ylim(-10,135)
		ax6.set_ylim(-5,45)

		# Labels.
		host.set_xlabel('$\mathrm{Grounding \ line  \ distance} \ (km)$',fontsize=22)
		ax.set_ylabel('$z_{\mathrm{bed}}$ $(km)$',fontsize=18, labelpad=7)
		ax2.set_ylabel('$u_{\mathrm{b}}$ $(km/yr)$', fontsize=18, labelpad=15)
		ax4.set_ylabel('$u_{\mathrm{def}}$ $(m/yr)$', fontsize=18, labelpad=15)
		ax3.set_ylabel('$z_{\mathrm{s}}$ $(km)$', fontsize=18, labelpad=2)
		ax6.set_ylabel('$N_{\mathrm{eff}}$ $(MPa)$', fontsize=18)
		ax7.set_ylabel('$\lambda$', fontsize=18)
		host.set_ylabel(r'$\tau_{b}$ $(kPa)$', fontsize=18)
		host.axis["left"].label.set_pad(8)
		host.axis["bottom"].label.set_pad(8)
		
		# Font size with axisartist.
		ax6.axis["right"].label.set_fontsize(18)
		ax7.axis["right"].label.set_fontsize(18)	
		host.axis["bottom"].label.set_fontsize(18)
		host.axis["left"].label.set_fontsize(18)
		
		# Ticks parameters.
		ax6.axis["right"].major_ticklabels.set(color="purple", size=15)
		ax7.axis["right"].major_ticklabels.set(color="darkred", size=15)
		host.axis["left"].major_ticklabels.set(color="black", size=15)
		host.axis["bottom"].major_ticklabels.set(color="black", size=15)
		
		#Ticks with axisartist.
		host.axis["bottom"].major_ticks.set_tick_out(True)
		host.axis["bottom"].major_ticks.set_ticksize(4)
		host.axis["bottom"].major_ticks.set_color("black")
		
		host.axis["left"].major_ticks.set_tick_out(True)
		host.axis["left"].major_ticks.set_ticksize(4)
		host.axis["left"].major_ticks.set_color("black")
		
		ax6.axis["right"].major_ticks.set_tick_out(True)
		ax6.axis["right"].major_ticks.set_ticksize(4)
		ax6.axis["right"].major_ticks.set_color("purple")
		
		ax7.axis["right"].major_ticks.set_tick_out(True)
		ax7.axis["right"].major_ticks.set_ticksize(4)
		ax7.axis["right"].major_ticks.set_color("darkred")
		
		
		
		
		# Tick labels.
		ax.set_xticks([d_plot[l_plot-1], -1200, -900, -600, -300, 0])
		ax2.set_xticks([-1500, -1200, -900, -600, -300, 0])
		ax7.set_xticks([-1500, -1200, -900, -600, -300, 0])
		ax7.set_xticklabels(['$-1500$', '$-1200$', '$-900$', '$-600$', '$-300$', '$0$'],fontsize=15)
		ax.set_yticks([-2.5, 0.0, 2.5, 5.0])
		ax.set_yticklabels(['$-2.5$', '$0.0$', '$2.5$', '$5.0$'],fontsize=15)
		ax2.set_yticks([0, 0.5,  1.0,  1.5])
		ax2.set_yticklabels(['$0$', '$0.5$', '$1.0$',  '$1.5$'],fontsize=15)
		ax3.set_yticks([-2.5, 0.0, 2.5, 5.0])
		ax3.set_yticklabels(['$-2.5$', '$0.0$', '$2.5$', '$5.0$'],fontsize=15)
		ax4.set_yticks([0, 25, 50, 75, 100])
		ax4.set_yticklabels(['$0$', '$25$', '$50$', '$75$', '$100$'],fontsize=15)
		host.set_yticks([0, 50, 100, 150, 200])
		host.set_yticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'],fontsize=15)
		ax6.set_yticks([0, 10, 20, 30, 40])
		ax6.set_yticklabels(['$0$', '$10$', '$20$', '$30$', '$40$'],fontsize=15)
		ax7.set_yticks([0.0, 0.5, 1.0])
		ax7.set_yticklabels(['$0.0$', '$0.5$', '$1.0$'],fontsize=15)
		

		ax.set_xticklabels([])
		ax2.set_xticklabels([])	


# 		host.set_xlabel('$\mathrm{Grounding \ line  \ distance} \ (km)$',fontsize=22)
# 		ax.set_ylabel('$z_{\mathrm{bed}}$ $(km)$',fontsize=18, labelpad=7)
# 		ax2.set_ylabel('$u_{\mathrm{b}}$ $(km/yr)$', fontsize=18, labelpad=15)
# 		ax4.set_ylabel('$u_{\mathrm{def}}$ $(m/yr)$', fontsize=18, labelpad=15)
# 		ax3.set_ylabel('$z_{\mathrm{s}}$ $(km)$', fontsize=18, labelpad=2)
# 		ax6.set_ylabel('$N_{\mathrm{eff}}$ $(MPa)$', fontsize=18)
# 		ax7.set_ylabel('$\lambda$', fontsize=18)
# 		host.set_ylabel(r'$\tau_{b}$ $(kPa)$', fontsize=18)

# 		# Label color.
# 		('brown')
# 		ax2.yaxis.label.set_color('blue')
# 		ax4.yaxis.label.set_color('blue')
# 		ax3.yaxis.label.set_color('darkgreen')
# 		host.yaxis.label.set_color('black')
# 		ax6.yaxis.label.set_color('purple')
# 		ax7.yaxis.label.set_color('darkred')
		ax.yaxis.label.set_color('brown')
		ax2.yaxis.label.set_color('blue')
		ax4.yaxis.label.set_color('blue')
		ax3.yaxis.label.set_color('darkgreen')
		host.yaxis.label.set_color('black')
		ax6.yaxis.label.set_color('purple')
		ax7.yaxis.label.set_color('darkred')
		
		
		# Tick parameters with ax.
		ax.tick_params(axis='y', which='major', length=4, color='brown')
		ax2.tick_params(axis='y', which='major', length=4, color='blue')
		ax3.tick_params(axis='y', which='major', length=4, color='darkgreen')
		ax4.tick_params(axis='y', which='major', length=4, color='blue')
		
		ax.tick_params(axis='y', colors='brown')
		ax2.tick_params(axis='y', colors='blue')
		ax4.tick_params(axis='y', colors='blue')
		ax3.tick_params(axis='y', colors='darkgreen')

		
		# Grid.
		ax.grid(axis='x', which='major', alpha=0.85)
		ax2.grid(axis='x', which='major', alpha=0.85)
		ax7.grid(axis='x', which='major', alpha=0.85)
		

		ax.invert_xaxis()
		ax2.invert_xaxis()
		ax7.invert_xaxis()


		##### Frame name ########
		if i<10:
 			frame = '00'+str(i)
		elif i>9 and i<100:
 			frame = '0'+str(i)
		else:
 			frame = str(i)
 			
		#plt.tight_layout()
		plt.savefig(path_save+'section_'+ensemble_name+'_'\
						  +str(frame)+'.png', bbox_inches='tight')
		plt.show()
		plt.close(fig)
		
 	
# SPATIAL DISTRIBUTION OF THE SECTION.
save_spatial = 0
if save_spatial == 1:
    
    x_tot = s[0] - j_read
    y_tot = k_read
    
    x_grnd = s[0] - j_grnd
    y_grnd = k_grnd
    
    N = 100*2
    		
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    
    colors_s = plt.cm.BrBG(np.linspace(0.3,0.0,N,endpoint=True))
    levels_s = np.concatenate([np.linspace(0,1,N+1)])
    cmap_s, norm_s = mcolors.from_levels_and_colors(levels_s, colors_s)
    
    
    ax.plot(x_tot,y_tot,linestyle='none', color='blue', marker= 'o', \
      markerfacecolor='blue', markersize=2, alpha=1.0, label=str(k_0)+str(j_0))
     	
    #ax.plot(x_grnd,y_grnd,linestyle='none', color='red', marker= 'x', \
     # markerfacecolor='none', markersize=8, alpha=1.0, label=str(k_0)+str(j_0))
    
    ax.contour(np.rot90(srf_0,3),np.linspace(0,0,1),linewidths=1.0,\
    	 linestyles='-',colors='black',norm=norm_s)
    
     	
    #	ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_title(r'Location of $(i_{0},j_{0})$ = $($'+str(k_0)+','+str(j_0)+'$)$',fontsize=16)
    #plt.savefig(path_save+'location_grnd_i_0_'+str(k_0)+'-j_0_'+str(j_0)+'.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)








