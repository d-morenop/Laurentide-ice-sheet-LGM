#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:33:40 2020

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


path_1   = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/PMIP3_sig50km/LIS-16KM_PMIP3-lgm-mean.nc'
path_2   = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_GHF-S04.nc'
path_top = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-RTOPO-2.0.1.nc'
path_ice = '/home/dmoren07/ice_data/Laurentide/LIS-16KM/LIS-16KM_TOPO-ICE-6G_C.nc'
path_fig = '/home/dmoren07/figures/boundary_conditions/'


varn      = ['pr_ann','t2m_ann','ghf','sftgif']
var_index = 0

		
# We load the initial surface elevation for visualization (land-ocean).
nc_PMIP3  = os.path.join(get_datadir(), path_1)
PMIP3     = Dataset(nc_PMIP3, mode='r')
pr_ann    = PMIP3.variables[varn[0]][:]
t2m       = PMIP3.variables[varn[1]][:]

nc_GEOTH  = os.path.join(get_datadir(), path_2)
GEOTH     = Dataset(nc_GEOTH, mode='r')
ghf       = GEOTH.variables[varn[2]][:]

nc_TOPOG  = os.path.join(get_datadir(), path_top)
TOPOG     = Dataset(nc_TOPOG, mode='r')
srf_0     = TOPOG.variables['z_srf'][:]
		
nc_ICE6G  = os.path.join(get_datadir(), path_ice)
ICE6G     = Dataset(nc_ICE6G, mode='r')
ice6G     = ICE6G.variables[varn[3]][0,:,:]	

#pr_ann = pr_ann / (24 * 10) # cm / h

N     = 100*2
N_int = np.int(N)

if var_index == 0:

	levels_s = np.concatenate([[0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.5]])
	colors_s =['#b3ffff','#5dd5d5','#2fb6b6',\
			   '#008fb3','#9933ff','#cc00ff','#8f00b3']
	cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap.set_over(plt.cm.jet(240))
	cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap_v.set_over(plt.cm.jet(240))
	
elif var_index == 1:

	levels_s = np.concatenate([[-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15]])
	colors =['#e88000','#ffb300','#fbff00','#ccff00','#84ff00','#00c227',\
		     '#00ffee','#59d6ff','#0091ff','#0055ff','#aa00ff']
	colors_s = colors[::-1]
	cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap.set_over(plt.cm.jet(240))
	cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap_v.set_over(plt.cm.jet(240))
	
elif var_index == 2:

	levels_s = np.concatenate([[40, 45, 50, 60, 65, 70, 80, 90, 100, 150]])
	colors_s =['#f9f2ec','#e6ccb3','#bf7d40','#86582d','#cc6600',\
		     '#ff8000','#ff5c33','#ff4d4d','#ff33cc']
	cmap, norm = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap.set_over(plt.cm.jet(240))
	cmap_v, norm_v = mcolors.from_levels_and_colors(levels_s, colors_s)
	cmap_v.set_over(plt.cm.jet(240))
	
		


###################################################################################
###################################################################################
#                                 FIGURES

fig = plt.figure(dpi=400)
ax  = fig.add_subplot(111)

plt.rcParams['text.usetex'] = True

cmap_s, norm_s = mcolors.from_levels_and_colors(levels_s, colors_s)

ax.contour(np.rot90(srf_0, 3), np.linspace(0, 0, 1), linewidths=1.0,\
			 linestyles='-', colors='black', norm=norm_s)	
ax.contour(np.rot90(ice6G, 3), np.linspace(50, 50, 1), linewidths=1.5,\
			 linestyles='-.', colors='red', norm=norm_s)
	
# We create new axes:  [left, bottom, width, height].

#cax = fig.add_axes([0.95, 0.04, 0.03, 0.92])
cax = fig.add_axes([0.92, 0.13, 0.03, 0.75])


if var_index == 0:
	im = ax.imshow(np.rot90(pr_ann,3), cmap=cmap_v, norm=norm_v)
	cb = fig.colorbar(im, cax=cax, extend='neither', pad=0.9)
	cb.set_ticks([0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.5])
	cb.set_ticklabels(['$0.01$', '$0.25$', '$0.50$', '$1.00$', '$2.00$', '$3.00$', \
								  '$4.00$', '$5.50$'])
	cb.ax.tick_params(labelsize=17)
	#cb.set_label(r'$ \mathrm{Annual \ prec.} \ (mm/d) $', rotation=270,\
	#		  labelpad=25, fontsize=22)

elif var_index == 1:
	im = ax.imshow(np.rot90(t2m,3), cmap=cmap_v, norm=norm_v)
	cb = fig.colorbar(im, cax=cax, extend='neither')
	cb.set_ticks([-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15])
	cb.set_ticklabels(['$-40$', '$-35$', '$-30$', '$-25$', '$-20$', '$-15$', \
								  '$-10$', '$-5$', '$0$', '$5$', '$10$', '$15$'])
	cb.ax.tick_params(labelsize=17)
	#cb.set_label(r'$ \mathrm{Annual \ mean\ T} \ (^{\circ} C) $', rotation=270,\
	#		  labelpad=25, fontsize=22)
	
elif var_index == 2:
	im = ax.imshow(np.rot90(ghf,3), cmap=cmap_v, norm=norm_v)
	cb = fig.colorbar(im, cax=cax, extend='neither')
	cb.set_ticks([40, 45, 50, 60, 65, 70, 80, 90, 100, 150])
	cb.set_ticklabels(['$40$', '$45$', '$50$', '$60$', '$65$', '$70$', \
								  '$80$', '$90$','$100$', '$150$'])
	cb.ax.tick_params(labelsize=17)
	#cb.set_label(r'$\mathrm{Geoth. \ heat \ flow} \ (mW / m^{2})$', \
	#		  rotation=270, labelpad=25, fontsize=22)

	
ax.invert_yaxis()
ax.set_yticklabels([])
ax.set_xticklabels([])
#plt.tight_layout()
plt.savefig(path_fig+varn[var_index]+'.png', bbox_inches='tight')
plt.show()
plt.close(fig)
	



