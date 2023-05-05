#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:44:45 2020

@author: dmoreno
"""

from __future__ import division
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from dimarray import read_nc, get_datadir
import os
plt.style.use("seaborn-white")



######################################################################
######################################################################

# Python script to plot time series from yelmo1D.nc.

######################################################################
######################################################################




path     = '/home/dmoren07/yelmo-model_v1.64/yelmox/output/LIS-32KM/'
path_fig = '/home/dmoren07/figures/v1.64/plots.1D/LIS-32KM/btmthd/'


ensemble = []

# List all subdirectories using os.listdir
ensemble = os.listdir(path)
ensemble.sort()

new = 0
if new == 1:
	os.makedirs(path_fig)



plot = np.array([2, 4, 6])
l  = len(plot)


for i in plot:
    nc_YELMO1D   = os.path.join(get_datadir(), path+ensemble[i]+'/yelmo1D.nc')
    data_YELMO1D = read_nc(nc_YELMO1D)
    YELMO1D      = Dataset(nc_YELMO1D, mode='r')
    t_yelmo      = 1.0e-3 * YELMO1D.variables['time'][:]



# Largest snapshot to be taken (ragarding simulations of different duration)
out_1D = 1                        # period of writing 1D data (kyr).


H     = []
H_max = []
A     = []
V     = []
V_f   = []
u_bar = []
H_w   = []
f_pmp = []
T_srf = []
dH_dt = []
dV_dt = []

for i in plot:
    print('Ensemble = '+ensemble[i])
    nc_YELMO1D   = os.path.join(get_datadir(), path+ensemble[i]+'/yelmo1D.nc')
    data_YELMO1D = read_nc(nc_YELMO1D)
    YELMO1D      = Dataset(nc_YELMO1D, mode='r')
    
    a = YELMO1D.variables['H_ice'][:]
    H.append(a)
    b = 1.0e-3 * YELMO1D.variables['H_ice_max'][:]
    H_max.append(b)
    c = YELMO1D.variables['V_sl'][:]
    V.append(c)
    d = YELMO1D.variables['A_ice'][:]
    A.append(d)
    e = YELMO1D.variables['H_w'][:]
    H_w.append(e)
    f = YELMO1D.variables['f_pmp'][:]
    f_pmp.append(f)
    g = YELMO1D.variables['T_srf'][:]
    T_srf.append(g)
    h = YELMO1D.variables['dHicedt'][:]
    dH_dt.append(h)
    m = YELMO1D.variables['dVicedt'][:]
    dV_dt.append(m)
    o = YELMO1D.variables['V_ice_f'][:]
    V_f.append(o)
    q = YELMO1D.variables['uxy_bar_g'][:]
    u_bar.append(q)

H = np.array(H)
H_max = np.array(H_max)
A = np.array(A)
V = np.array(V)
V_f = np.array(V_f)
u_bar = np.array(u_bar)
H_w = np.array(H_w)
f_pmp = np.array(f_pmp)
T_srf = np.array(T_srf)
dH_dt = np.array(dH_dt)
dV_dt = np.array(dV_dt)



t_plot = []

# for i in plot:
#     t_vector = np.linspace(0, out_1D*(t[i]-1), t[i])
#     t_plot.append(t_vector)

# t_plot = np.array(t_plot)

color  = ['black','red','blue']
marker = ['o','o','o','o']
legend =  [r'$\mathrm{Linear}$', \
		   r'$\mathrm{Purely \ plastic}$',\
		   r'$\mathrm{Coluomb}$']


###################################################################################
###################################################################################
#                                 H_ICE

# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],H[i], linestyle='-.', color=color[i],\
# 			  marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='lower right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Averaged ice thickness (m)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(2300, 2701, 50)
# minor_ticks_y = np.arange(2300, 2701, 25)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('$\overline{H}_{ice}$ - '+str(exp_name))
# plt.savefig(str(path)+'H_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



###################################################################################
###################################################################################
#                                 V_sl


fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)

plt.rcParams['text.usetex'] = True

for i in range(l):
    plt.plot(t_yelmo, V[i], linestyle='-', color=color[i], \
			 marker='None', linewidth=3.0, markersize=4, alpha=1.0, label=legend[i])
    
plt.legend(loc='lower right', ncol = 1, frameon = True, \
		   framealpha = 1.0, fontsize = 12, fancybox = True)

ax.set_xlabel(r'$\mathrm{Time} \ (kyr)$', fontsize=18)
ax.set_ylabel(r'$V_{sl} \ (10^{6} km^{3})$', fontsize=18)


ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], \
 							      fontsize=15)
ax.set_yticks([30, 32, 34, 36, 38])
ax.set_yticklabels(['$30$', '$32$', '$34$', '$36$', '$38$'], \
 							     fontsize=15)

ax.tick_params(axis='both', which='major', length=4, colors='black')

ax.set_xlim(0, 200) 
	
plt.tight_layout()
#ax.title.set_text(r'$V_{ice}$')
plt.savefig(path_fig+'V_sl_btmthd.cffrzn.0.10.png', bbox_inches='tight')
plt.show()


###################################################################################
###################################################################################
#                                 V_f


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],V_f[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='lower right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Floating ice volume (10$^{6}$ Km$^{3}$)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(0.10, 0.30, 0.025)
# minor_ticks_y = np.arange(0.10, 0.30, 0.0125)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('V$_{f,ice}$ - '+str(exp_name))
# plt.savefig(str(path)+'V_f_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



# ###################################################################################
# ###################################################################################
# #                                 uxy_bar


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],u_bar[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    

# plt.legend(loc='top right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Mean depth-averaged velocity (m/yr)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(20, 66, 5)
# minor_ticks_y = np.arange(20, 66, 2.5)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('$\overline{u}$ - '+str(exp_name))
# plt.savefig(str(path)+'u_bar_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()


# ###################################################################################
# ###################################################################################
# #                                 A_ICE


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],A[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='lower right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Ice area (10$^{6}$ Km$^{2}$)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(20, 22, 0.25)
# minor_ticks_y = np.arange(20, 22, 0.1)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('A$_{ice}$ - '+str(exp_name))
# plt.savefig(str(path)+'A_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



###################################################################################
###################################################################################
#                                 H_ICE_max


fig = plt.figure(dpi=400)
ax = fig.add_subplot(111)

plt.rcParams['text.usetex'] = True

for i in range(l):
    plt.plot(t_yelmo, H_max[i], linestyle='-', color=color[i], \
			 marker='None', linewidth=3.0, markersize=4, alpha=1.0, label=legend[i])
    
plt.legend(loc='upper right', ncol = 1, frameon = True, \
		   framealpha = 1.0, fontsize = 12, fancybox = True)

ax.set_xlabel(r'$\mathrm{Time} \ (kyr)$', fontsize=18)
ax.set_ylabel(r'$H_{max} \ (km)$', fontsize=18)


ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels(['$0$', '$50$', '$100$', '$150$', '$200$'], \
 							      fontsize=15)
ax.set_yticks([3.5, 3.75, 4.0, 4.25, 4.5])
ax.set_yticklabels(['$3.50$', '$3.75$', '$4.0$', '$4.25$', '$4.50$'], fontsize=15)

ax.tick_params(axis='both', which='major', length=4, colors='black')

ax.set_xlim(0, 200) 
	
plt.tight_layout()
#ax.title.set_text(r'$H_{max}$')
plt.savefig(path_fig+'H_max_btmthd.cffrzn.0.10.png', bbox_inches='tight')
plt.show()



###################################################################################
###################################################################################
#                                 H_w


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],H_w[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])

# plt.legend(loc='lower right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Mean basal water thickness (m)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(0, 0.32, 0.02)
# minor_ticks_y = np.arange(0, 0.32, 0.01)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('$\overline{H}_{w}$ - '+str(exp_name))
# plt.savefig(str(path)+'H_w_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



# ###################################################################################
# ###################################################################################
# #                                 f_pmp


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],f_pmp[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='top right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Grid fraction at melt point')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(0, 0.3, 0.05)
# minor_ticks_y = np.arange(0, 0.3, 0.025)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('f$_{pmp}$ - '+str(exp_name))
# plt.savefig(str(path)+'f_pmp_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



# ###################################################################################
# ###################################################################################
# #                                 T_sfr


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],T_srf[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])

# plt.legend(loc='top right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('Mean surface temperature (K)')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(214, 217, 0.5)
# minor_ticks_y = np.arange(214, 217, 0.25)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('T$_{srf}$ - '+str(exp_name))
# plt.savefig(str(path)+'T_srf_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()



# ###################################################################################
# ###################################################################################
# #                                 dH_dt


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],dH_dt[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='top right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('$dH/dt$ $(m/yr)$')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(0, 0.29, 0.04)
# minor_ticks_y = np.arange(0, 0.29, 0.02)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('$dH/dt$ - '+str(exp_name))
# plt.savefig(str(path)+'dHdt_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()





# ###################################################################################
# ###################################################################################
# #                                 dV_dt


# fig = plt.figure(dpi=400)
# ax = fig.add_subplot(111)

# for i in plot:
#     plt.plot(t_plot[i],dV_dt[i], linestyle='-.', color=color[i], marker=marker[i], linewidth=0.3, markersize=4, alpha=1.0, label=title_name[i])
    
# plt.legend(loc='top right', ncol = 1, frameon = True, framealpha = 1.0, fontsize = 10, fancybox = True)
# plt.xlabel('Time (10$^{3}$ yr)')
# plt.ylabel('$dV/dt$ $(km^{3}/yr)$')


# # Grid settings.
# major_ticks_x = np.arange(0, out_1D*t_max, 25)
# minor_ticks_x = np.arange(0, out_1D*t_max, 12.5)
# major_ticks_y = np.arange(0, 6001, 500)
# minor_ticks_y = np.arange(0, 6001, 250)
# ax.set_xticks(major_ticks_x)
# ax.set_xticks(minor_ticks_x, minor=True)
# ax.set_yticks(major_ticks_y)
# ax.set_yticks(minor_ticks_y, minor=True)
# ax.grid(which='both')
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

# plt.tight_layout()
# ax.title.set_text('$dV/dt$ - '+str(exp_name))
# plt.savefig(str(path)+'dVdt_'+str(exp_name)+'_'+str(ensemble[0])+'.png', bbox_inches='tight')
# plt.show()





