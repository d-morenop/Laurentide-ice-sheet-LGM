#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:46:49 2020

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


######################################################################
######################################################################

# PYTHON SCRIPT TO READ AND PLOT .nc FILES FROM YELMO.
# Here we create time series plots given a fixed point (x0,y0). 
# Thus, we can compare different variables at each timestep over
# a certain location (e.g., narrow streams, Hudson Bay, etc).

######################################################################
######################################################################



#ensemble_1 = ['btmthd.1.btq.0.1','btmthd.1.btq.0.3','btmthd.1.btq.1.0','btmthd.2.btq.0.1','btmthd.2.btq.0.3','btmthd.2.btq.1.0']
#ensemble_2 = ['btmthd.2.btq.0.1','btmthd.2.btq.0.3','btmthd.2.btq.1.0','btmthd.3.btq.0.1','btmthd.3.btq.0.3','btmthd.3.btq.1.0']

ensemble_2 = ['cffrzn.0.05.cfstrm.0.005','cffrzn.0.05.cfstrm.0.01','cffrzn.0.05.cfstrm.0.02',\
              'cffrzn.0.10.cfstrm.0.005','cffrzn.0.10.cfstrm.0.01','cffrzn.0.10.cfstrm.0.02',\
              'cffrzn.0.15.cfstrm.0.005','cffrzn.0.15.cfstrm.0.01','cffrzn.0.15.cfstrm.0.02']

exp = ['exp_0','exp_1','exp_2','exp_6','exp_7','v0.99']
exp_name = exp[5]

varn = ['H_ice','visc_eff','f_pmp','H_w','uxy_bar','beta','Q_b']
n = len(varn)

# We load the initial surface elevation for visualization (land-ocean).
nc_YELMO2D  = os.path.join(get_datadir(), '/home/dmoreno/yelmo-model/yelmox/output/PMIP3/'+str(exp_name)+'/yelmo2D_cffrzn.0.15.cfstrm.0.02.nc')
YELMO2D = Dataset(nc_YELMO2D, mode='r')
srf_0 = YELMO2D.variables['z_srf'][0,:,:]

# Fixed (x,y) point.
# Since axis are inverted and coordinate origin has changed, we write an expression
# to relate the point in the python plot (x0, y0) and the .nc grid point (i0, j0) given by ncview.
s = np.shape(srf_0)
i_0 = 165
j_0 = 15
x_0 = s[1] - i_0
y_0 = s[0] - j_0

out_2D = 5                     # period of writing 2D data (kyr).

l = len(ensemble_2)
t = np.empty(l) # vector with values of temporal dimensions
Q_sum = np.empty(l)
max_array = np.empty(l) # vector with highest varaible value


col_1 = np.linspace(1,0,l+1) # color = (red, green, blue)
col_2 = np.linspace(0,1,l+1) # color = (red, green, blue)



for k in range(0,n-1):

    var_index = k
    var_name = varn[var_index]
    
    var_all = []
    time_plt = []
    
    for i in range(l):
        nc_YELMO2D  = os.path.join(get_datadir(), '/home/dmoreno/yelmo-model/yelmox/output/PMIP3/'+str(exp_name)+'/yelmo2D_'+str(ensemble_2[i])+'.nc')
        YELMO2D = Dataset(nc_YELMO2D, mode='r')
        var = YELMO2D.variables['time'][:]
        
        time = np.size(var)
        print('t = '+str(time))
        t[i] = time
        t_plot = np.linspace(0,t[i]-1,t[i])
        time_plt.append(t_plot)
    
    t_plt = np.array(time_plt)
    # Largest snapshot to be taken (ragarding simulations of different duration)
    max_frame = np.min(t)
    max_time = np.max(t)
    n = np.floor(max_frame)-1.0
    n = np.int_(n)
    print('n = '+str(n))
    
   
#    t_plot[i] = i*out_2D                    # time in kyr
    t_max = max_time + 1.0
    
    
    for i in range(l):
        nc_YELMO2D  = os.path.join(get_datadir(), '/home/dmoreno/yelmo-model/yelmox/output/PMIP3/'+str(exp_name)+'/yelmo2D_'+str(ensemble_2[i])+'.nc')
        YELMO2D = Dataset(nc_YELMO2D, mode='r')
        w = YELMO2D.variables[str(var_name)][:,y_0,x_0]
    
        if var_index==1:
            for j in range(len(w)):
                    if w[j]==0.0:
                        w[j] = np.nan
                        
            w = np.log10(w)
    
        var_all.append(w)
        max_array[i] = np.max(w)
    
    var_max = np.max(max_array)
    var_all = np.array(var_all)

    
    
        
    time_plt = np.array(time_plt)
    
    
    
    
    ###################################################################################
    ###################################################################################
    #                                 PLOT
    
#    label = ['Power-law, q=0.1','Power-law, q=1/3','Power-law, q=1.0','Coulomb, q=0.1','Coulomb, q=1/3','Coulomb, q=1.0']
    label = ['C$_{frz}$ = 0.05, C$_{strm}$ = 0.005','C$_{frz}$ = 0.05, C$_{strm}$ = 0.01','C$_{frz}$ = 0.05, C$_{strm}$ = 0.02',\
             'C$_{frz}$ = 0.10, C$_{strm}$ = 0.005','C$_{frz}$ = 0.10, C$_{strm}$ = 0.01','C$_{frz}$ = 0.10, C$_{strm}$ = 0.02',\
             'C$_{frz}$ = 0.15, C$_{strm}$ = 0.005','C$_{frz}$ = 0.15, C$_{strm}$ = 0.01','C$_{frz}$ = 0.15, C$_{strm}$ = 0.02']
    
    marker = ['o','o','o','o','o','o','o','o','o']
#    color = ['blue','red','black','blue','red','black']
#    marker = ['o','o','o','v','v','v']
    ylabel = ['Averaged ice thickness','Effective viscosity','f$_{pmp}$','Mean basal water thickness','Vertically integrated velocity','Basal friction coeff.','Basal frictional heating']
    title =['H$_{ice}$ (m)','$\eta$ (Pa yr m)','f$_{pmp}$','H$_{w}$ (m)','$\overline{u}_{xy}$ (m/yr)','Beta (Pa yr m$^{-1}$)','Q$_{b}$ (J yr$^{-1}$ m$^{-2}$)']
    
    fig = plt.figure(dpi=400)
    ax = fig.add_subplot(111)
    
    for i in range(4,5):
    
        plt.plot(t_plt[i],var_all[i], linestyle='-.', color=(col_1[i],0,col_2[i]), marker= marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=label[i])
        
    plt.legend(loc='lower right', ncol = 2, frameon = True, framealpha = 1.0, fontsize = 8, fancybox = True)
    plt.xlabel('Time (10$^{3}$ yr)')
    plt.ylabel(str(ylabel[var_index]))
    
    # Grid settings.
    if var_max<10:
        y_length = 0.5
    elif var_max>9 and var_max<20:
        y_length = 1.0
    elif var_max>20 and  var_max<40:
        y_length = 5.0
    elif var_max>40 and var_max<100:
        y_length = 10.0
    elif var_max>100 and var_max<1000:
        y_length = 50.0
    elif var_max>3000 and var_max<7000:
        y_length = 500.0
    elif var_max>1e5:
        y_length = 1e4
    elif var_max>1e8:
        y_length = 1e7
    elif var_max==1:
        y_length=0.1
        
    major_ticks_x = np.arange(0, t_max, 10)
    minor_ticks_x = np.arange(0, t_max, 5)
    major_ticks_y = np.arange(0, 1.1*var_max, y_length)            #1e4 for Q_b
    minor_ticks_y = np.arange(0, 1.1*var_max, 0.5*y_length)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    
    plt.tight_layout()
    ax.title.set_text(str(title[var_index])+' '+str(exp_name))
    path = 'figures/'+str(exp_name)+'/time_series/fixed_x0-y0/'
    plt.savefig(str(path)+str(var_name)+'_'+str(exp_name)+'_'+str(ensemble_2[0])+'.png', bbox_inches='tight')
    plt.show()
    
    
    
    
    #################################################################
    #################################################################
    
# Representation of the chosen grid point on the map.
    
max_z = np.max(srf_0)
    
N = 100*2
lim_min = 0
lim_med_1 = 0.4*max_z
lim_med_2 = 0.6*max_z
lim_max = max_z
colors1 = plt.cm.seismic(np.linspace(0.5, 0.1, 0.4*N))
colors2 = plt.cm.plasma(np.linspace(0.02, 1.0, 0.2*N))
colors3 = plt.cm.autumn(np.linspace(1.0, 0.0, 0.4*N))
levels = np.concatenate([np.linspace(lim_min,lim_med_1,0.4*N+1),\
                         np.linspace(lim_med_1,lim_med_2,0.2*N),\
                         np.linspace(lim_med_2,lim_max,0.4*N)])
colors = np.vstack((colors1,colors2,colors3))
cmap_v, norm_v = mcolors.from_levels_and_colors(levels, colors)
cmap_v.set_over(plt.cm.RdYlBu(0))
    
colors_s = plt.cm.BrBG(np.linspace(0.3,0.0,N,endpoint=True))
levels_s = np.concatenate([np.linspace(lim_min,lim_max,N+1)])
cmap_s, norm_s = mcolors.from_levels_and_colors(levels_s, colors_s)
    
    
    
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)

im = ax.imshow(np.rot90(np.rot90(np.rot90(srf_0))),cmap=cmap_v,norm=norm_v)
    
ax.plot(y_0,x_0,linestyle='none', color='red', marker= 'x', markersize=6, alpha=1.0, label=str(i_0)+str(j_0))
ax.contour(np.rot90(np.rot90(np.rot90(srf_0))),np.linspace(0,0,1),linewidths=1.0,linestyles='-',colors='black',norm=norm_s)
    
cax = fig.add_axes([0.95, 0.05, 0.03, 0.9])
ticks = np.linspace(lim_min,np.rint(lim_max),10)
    
cb = fig.colorbar(im, cax=cax,extend='max',ticks=ticks)
cb.set_ticks([0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000]) #H_ice
cb.set_label('Z$_{srf}$ (m)', rotation=270,labelpad=15,fontsize=12)
    
ax.invert_yaxis()
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.tight_layout()
ax.title.set_text('Location of (i$_{0}$,j$_{0}$)')
path = 'figures/'+str(exp_name)+'/time_series/fixed_x0-y0/'
plt.savefig(str(path)+'i0_'+str(i_0)+'-j0_'+str(j_0)+'_'+str(exp_name)+'.png', bbox_inches='tight')
plt.show()
    
    









