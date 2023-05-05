#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:33:25 2023

@author: dmoreno

Script to plot LIS reconstruction from Stokes (2017)
"""

import numpy as np
import matplotlib.pyplot as plt


path_fig  = '/home/dmoreno/figures/paper.a/stokes_2017/'

plot_legend = True
save_fig    = True


# Vector with studies.
x_stokes = [r'$ \mathrm{Ramsay \ (1931)} $', \
     r'$ \mathrm{Donn \ et \ al. \ (1962)} $', \
     r'$ \mathrm{Andrews \ (1969)} $', \
     r'$ \mathrm{Flint \ (1971)} $', \
     r'$ \mathrm{Paterson \ (1972)} $', \
     r'$ \mathrm{Sugden \ (1977)} $', \
     r'$ \mathrm{Budd \ \& \ Smith \ (1981)} $', \
     r'$ \mathrm{Boulton \ et \ al. \ (1985)} $', \
     r'$ \mathrm{Fisher \ et \ al. \ (1985)} $', \
     r'$ \mathrm{Tushingham \ \& \ Peltier \ (1991) \ (ICE3G)} $', \
     r'$ \mathrm{Peltier \ (1994) \ (ICE4G) } $', \
     r'$ \mathrm{Clark \ et \ al. \ (1996)} $', \
     r'$ \mathrm{Ramsay \ (1931)} $', \
     r'$ \mathrm{Marshall \ \& \ Clark \ (1997a,b)} $', \
     r'$ \mathrm{Licciardi \ et \ al. \ (1998)} $', \
     r'$ \mathrm{Tarasov \ \& \ Peltier \ (1999)} $', \
     r'$ \mathrm{Peltier \ (2004) \ (ICE5G) } $', \
     r'$ \mathrm{Andrews \ (2006)} $', \
     r'$ \mathrm{Tarasov \ et \ al. \ (2012)} $', \
     r'$ \mathrm{Gregoire \ et \ al. \ (2012)} $', \
     r'$ \mathrm{Lambeck \ et \ al. \ (2017)} $', \
     r'$ \mathrm{This \ study} $']

x = [r'$ \mathbf{a} $', \
     r'$ \mathbf{b} $', \
     r'$ \mathbf{c} $', \
     r'$ \mathbf{d} $', \
     r'$ \mathbf{e} $', \
     r'$ \mathbf{f} $', \
     r'$ \mathbf{g} $', \
     r'$ \mathbf{h} $', \
     r'$ \mathbf{i} $', \
     r'$ \mathbf{j} $', \
     r'$ \mathbf{k} $', \
     r'$ \mathbf{l} $', \
     r'$ \mathbf{m} $', \
     r'$ \mathbf{n} $', \
     r'$ \mathbf{o} $', \
     r'$ \mathbf{p} $', \
     r'$ \mathbf{q} $', \
     r'$ \mathbf{r} $', \
     r'$ \mathbf{s} $', \
     r'$ \mathbf{t} $', \
     r'$ \mathbf{u} $', \
     r'$ \mathbf{v} $']

x_num = [r'$ \mathbf{1} $', \
     r'$ \mathbf{2} $', \
     r'$ \mathbf{3} $', \
     r'$ \mathbf{4} $', \
     r'$ \mathbf{5} $', \
     r'$ \mathbf{6} $', \
     r'$ \mathbf{7} $', \
     r'$ \mathbf{8} $', \
     r'$ \mathbf{9} $', \
     r'$ \mathbf{10} $', \
     r'$ \mathbf{11} $', \
     r'$ \mathbf{12} $', \
     r'$ \mathbf{13} $', \
     r'$ \mathbf{14} $', \
     r'$ \mathbf{15} $', \
     r'$ \mathbf{16} $', \
     r'$ \mathbf{17} $', \
     r'$ \mathbf{18} $', \
     r'$ \mathbf{19} $', \
     r'$ \mathbf{20} $', \
     r'$ \mathbf{21} $']

A = np.array([15.75, 
              12.74, 
              11.81,
              13.39,
              11.6,
              np.nan,
              np.nan,
              np.nan,
              np.nan,
              np.nan,
              np.nan,
              np.nan,
              np.nan,
              14,
              np.nan,
              13,
              np.nan,
              12,
              np.nan,
              16,
              np.nan])

H_max = np.array([2.9, 
                  np.nan,
                  np.nan,
                  np.nan,
                  2.7,
                  3.5,
                  4.25,
                  3.65,
                  3.25,
                  3.0,
                  3,
                  3,
                  2.25,
                  4.2,
                  3.35,
                  3.8,
                  4,
                  3.5,
                  np.nan,
                  3,
                  3.5])


V = np.array([45.45,
              0.5*(31.85+25.48),
              26,
              29.46,
              26.5,
              37.0,
              32,
              0.5*(34.2+30.5),
              0.5*(44+33),
              18,
              21,
              19,
              19.7,
              36.4,
              0.5*(19.7+15.9),
              25,
              np.nan,
              np.nan,
              28,
              35,
              np.nan])


H_max_err = np.array([np.nan,
                      np.nan,
                      np.nan,
                      np.nan,
                      np.nan,
                      np.nan,
                      0.25,
                      0.15,
                      0.25,
                      0.2,
                      np.nan,
                      0.0,
                      0.25,
                      np.nan,
                      0.25,
                      np.nan,
                      0.0,
                      0.5,
                      np.nan,
                      0.0,
                      0.0])



V_err = np.array([np.nan,
                31.85-25.48,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.0,
                34.2-30.5,
                44-33,
                25.9-21.1,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                19.7-15.9,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan])


# Dimensions.
l = len(A)

# Moreno-Parada et al. (2022).
x_moreno = np.full(l+1, np.nan)
A_moreno = np.full(l+1, np.nan)
H_moreno = np.full(l+1, np.nan)
V_moreno = np.full(l+1, np.nan)
x_moreno[l] = l
A_moreno[l] = 16.0
H_moreno[l] = 4.15
V_moreno[l] = 33.5



# Positions where error is assymmetric.
H_uplim = np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1], dtype=bool)

# Set lower and upper errors.
H_low_err = H_max_err
H_up_err  = H_max_err
H_up_err  = np.where(H_uplim == False, H_up_err, 100.0)

# Assymetric error
H_error = np.array(list(zip(H_low_err, H_up_err))).T



# Solf bed model Boulton el al. and Fisher et al.
x_soft = np.full(l, np.nan)
H_soft = np.full(l, np.nan)

soft_low_err   = np.zeros(l)
soft_up_err    = np.zeros(l)

x_soft[8] = 8
x_soft[9] = 9
H_soft[8] = 3.0
H_soft[9] = 3.2
soft_up_err[8] = 100.0
soft_up_err[9] = 100.0

error_soft = np.array(list(zip(soft_low_err, soft_up_err))).T



# Volume error.
V_lolim = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=bool)

V_low_err = V_err
V_up_err  = V_err
V_low_err = np.where(V_lolim == False, V_low_err, 100.0)

# Assymetric error
V_error = np.array(list(zip(V_low_err, V_up_err))).T




# Figure.
fig = plt.figure(dpi=1000, figsize=(8,6))
ax  = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

x_plot  = np.arange(0, l, 1)
x_ticks = np.arange(0, len(x_stokes), 1)

plt.rcParams['text.usetex'] = True


ax.plot(x_plot, A, linestyle='None', color='red', marker='o', \
        markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 

ax2.errorbar(x_plot, H_max, yerr=H_error, linestyle='None', \
             color='black', marker='o', markersize=7.0, linewidth=1.5, \
             alpha=1.0, label=r'$u_{b}(x)$', capsize=5) 

#Plot soft/hard bed bounds.
ax2.errorbar(x_soft, H_soft, yerr=error_soft, linestyle='None', \
             color='magenta', marker='o', markersize=7.0, linewidth=1.5, \
             alpha=1.0, label=r'$u_{b}(x)$', capsize=5) 

ax3.errorbar(x_plot, V, yerr=V_error, linestyle='None', \
             color='blue', marker='o', markersize=7.0, linewidth=1.5, \
             alpha=1.0, label=r'$u_{b}(x)$', capsize=5) 

# Plot Moreno-Parada et al. (2022).
ax.plot(x_moreno, A_moreno, linestyle='None', color='red', marker='^', \
        markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
ax2.plot(x_moreno, H_moreno, linestyle='None', color='black', marker='^', \
        markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 
ax3.plot(x_moreno, V_moreno, linestyle='None', color='blue', marker='^', \
        markersize=7.0, linewidth=2.5, alpha=1.0, label=r'$u_{b}(x)$') 


# Inset axis with boxplot.
#ax_inset  = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
ax_inset  = ax.inset_axes([1.05, 0, 0.25, 1])
ax2_inset = ax2.inset_axes([1.05, 0, 0.25, 1])
ax3_inset = ax3.inset_axes([1.05, 0, 0.25, 1])


ax_inset.yaxis.tick_right()
ax2_inset.yaxis.tick_right()
ax3_inset.yaxis.tick_right()

# Remove nan entries to plot.
A     = A[~np.isnan(A)]
H_max = H_max[~np.isnan(H_max)]
V     = V[~np.isnan(V)]

"""
vp1 = ax_inset.violinplot(A, points=60, widths=0.7,
                     showmeans=False, showextrema=False, showmedians=False,
                     quantiles=[0.1, 0.25, 0.75, 0.9], bw_method=0.5)
"""

vp1 = ax_inset.violinplot(A, points=60, widths=0.7,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.5)

vp2 = ax2_inset.violinplot(H_max, points=60, widths=0.7,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.5)

vp3 = ax3_inset.violinplot(V, points=60, widths=0.7,
                     showmeans=False, showextrema=False, showmedians=False,
                     bw_method=0.5)


colours   = ['red', 'grey', 'blue']
col_edges = ['brown', 'black', 'navy']
vps       = [vp1, vp2, vp3]

for vp, colour, col_edges in zip(vps, colours, col_edges):
        for pc in vp['bodies']:
                pc.set_facecolor(colour)
                pc.set_edgecolor(col_edges)
                pc.set_alpha(0.4)


data = [A, H_max, V]
axis = [ax_inset, ax2_inset, ax3_inset]


stat_col = ['red', 'black', 'blue']

for data_now, ax_now, col_now in zip(data, axis, stat_col):
        q1, q2, medians, q3, q4 = np.percentile(data_now, [10, 25, 50, 75, 90], axis=0)

        inds = np.arange(1, 2)
        ax_now.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax_now.vlines(inds, q2, q3, color=col_now, linestyle='-', alpha=1.0, lw=5, zorder=2)
        ax_now.vlines(inds, q1, q4, color=col_now, linestyle='-', alpha=1.0, lw=2, zorder=2)



ax.set_ylabel(r'$ A \ (10^6 \ \mathrm{km}^2)$', fontsize=19)
ax2.set_ylabel(r'$z_{\mathrm{max}} \ (\mathrm{km})$', fontsize=19)
ax3.set_ylabel(r'$ V \ (10^6 \ \mathrm{km}^3)$', fontsize=19)
#ax.set_xlabel(r'$H_{gl} \ (km)$', fontsize=18)

ax.set_xlim(-0.5, 21.5)
ax2.set_xlim(-0.5, 21.5)
ax3.set_xlim(-0.5, 21.5)
ax.set_ylim(10, 16.5)
ax2.set_ylim(2, 5)
ax3.set_ylim(10, 50)
    
ax.yaxis.label.set_color('black')

ax.set_yticks([10, 12, 14, 16])
ax.set_yticklabels(['$10$', '$12$', '$14$', '$16$'], fontsize=15)

ax2.set_yticks([2, 3, 4, 5])
ax2.set_yticklabels(['$2.0$', '$3.0$', '$4.0$', '$5.0$'], fontsize=15)

ax3.set_yticks([10, 20, 30, 40, 50])
ax3.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], fontsize=15)

ax.set_xticks(x_ticks)
ax2.set_xticks(x_ticks)
ax3.set_xticks(x_ticks)

ax.set_xticklabels([])
ax2.set_xticklabels([])
#ax3.set_xticklabels(x, rotation = 60, fontsize=10)
ax3.set_xticklabels(x, rotation = 0, fontsize=17)

ax.tick_params(axis='x', which='major', length=0, colors='black')
ax2.tick_params(axis='x', which='major', length=0, colors='black')
ax3.tick_params(axis='both', which='major', length=4, colors='black')

ax.grid(axis='x', which='major', alpha=0.85)
ax2.grid(axis='x', which='major', alpha=0.85)
ax3.grid(axis='x', which='major', alpha=0.85)


# Inset settings.
ax_inset.set_xticklabels([])
ax2_inset.set_xticklabels([])
ax3_inset.set_xticklabels([])

ax_inset.set_yticklabels([])
ax2_inset.set_yticklabels([])
ax3_inset.set_yticklabels([])

ax_inset.tick_params(axis='both', which='major', length=0, colors='black')
ax2_inset.tick_params(axis='both', which='major', length=0, colors='black')
ax3_inset.tick_params(axis='both', which='major', length=0, colors='black')


plt.tight_layout()

if save_fig == True:
    plt.savefig(path_fig+'stokes_2017.png', bbox_inches='tight')

# Display and close figure.
plt.show()
plt.close(fig)



if plot_legend == True:
        # PLOT FOR THE LEGEND IN X AXIS.
        fig = plt.figure(dpi=600, figsize=(8,6))
        ax = fig.add_subplot(111)

        #fig.patch.set_visible(False)
        ax.axis('off')

        x_nan = np.full(len(x), np.nan)

        plt.rcParams['text.usetex'] = True

        for i in range(len(x)):
                ax.plot(x_nan[i], linestyle='None', color='black', marker=x[i], \
                        markersize=7.0, linewidth=2.5, alpha=1.0, label=x_stokes[i]) 

        ax.legend(loc = "upper center", title = r'$ \mathbf{References} $', \
                title_fontsize = 15, frameon = True, framealpha = 1.0, \
                fontsize = 13, fancybox = True, ncol=1)


        if save_fig == True:
                plt.savefig(path_fig+'legend.png', bbox_inches='tight')

        plt.show()
        plt.close(fig)