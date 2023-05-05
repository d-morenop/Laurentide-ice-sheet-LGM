#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  66 14:35:49 2023

@author: dmoreno
"""


import os
import numpy as np
from netCDF4 import Dataset
from dimarray import get_datadir
import matplotlib as mpl
import matplotlib.pyplot as plt

import geopandas as gpd
import pyproj
from shapely.geometry import Polygon


path = "/home/dmoren07/ice_data/Margold-et-al_2014/"
path_fig = "/home/dmoren07/figures/paper.a/margold-2014_ice-streams/"

file_flowlines       = "IS_flowline.shp"
file_polygons        = "IS_polygons.shp"
file_polygons_polar  = "IS_polygons_polar-stereographic.shp"



# Define the input and output coordinate systems.
input_crs  = pyproj.CRS('EPSG:3978')
output_crs = pyproj.CRS('+proj=stere +lat_0=70 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')


# Read files with geopandas
dataframe_flowlines = gpd.read_file(path+file_flowlines, crs=input_crs)
dataframe_polygons  = gpd.read_file(path+file_polygons, crs=input_crs)

#print(dataframe.head())
#print(dataframe.columns)

geometry = dataframe_flowlines["geometry"]
print('Crs polygons = ', dataframe_polygons["geometry"].crs)
print('Crs flowlines = ', dataframe_flowlines["geometry"].crs)



# Transfor to yelmo LIS coordindates:
# Load the input data and reproject it to the output coordinate system
reprojected_dataframe = dataframe_polygons.to_crs(output_crs)

# Save the reprojected data to a new file
reprojected_dataframe.to_file(path+file_polygons_polar)

dataframe_polygons_polar = gpd.read_file(path+file_polygons_polar)
#print('Crs polygons polar = ', dataframe_polygons_polar["geometry"].crs)





# Flowlines plot
plt.rcParams['text.usetex'] = True
ax = dataframe_flowlines.plot(column="geometry")
ax.figure.set_dpi(800)
ax.set_title(r"$ \mathrm{Margold \ et \ al. \ (2014)}$")
ax.tick_params(axis='both', which='major', length=0, colors='red')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.tight_layout()

#ax.figure.savefig(path_fig+"margold-2014_flowlines.png", dpi=800)
plt.show()
plt.close()


# Polygons plot.
ax2 = dataframe_polygons.plot(column="geometry")
ax2.figure.set_dpi(800)
ax2.set_title(r"$ \mathrm{Margold \ et \ al. \ (2014)}$")
ax2.tick_params(axis='both', which='major', length=0, colors='red')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.tight_layout()
#ax2.figure.savefig(path_fig+"margold-2014_polygons.png", dpi=800)

plt.show()
plt.close()

# Plot with transformed cordinate system.
ax3 = dataframe_polygons_polar.plot(column="geometry")
ax3.figure.set_dpi(800)
ax3.set_title(r"$ \mathrm{Polar \ - \ Margold \ et \ al. \ (2014)}$")
ax3.tick_params(axis='both', which='major', length=0, colors='red')
ax3.set_xticklabels([])
ax3.set_yticklabels([])

ax3 = dataframe_polygons_polar.plot(column="geometry")

plt.tight_layout()
#ax3.figure.savefig(path_fig+"margold-2014_polygons.png", dpi=800)

plt.show()
plt.close()


# Convert numpy array to dataframe to plot them in the same panel

# create a geopandas dataframe from the numpy array
crs     = dataframe_polygons_polar.crs # get the CRS from the given dataframe
polygon = Polygon(strm_mask)
gdf     = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon])

<<<<<<< HEAD
# plot the two dataframes in the same panel
ax4 = dataframe_polygons_polar.plot(figsize=(10, 10))
gdf.plot(ax=ax, facecolor='none', edgecolor='red')

# add legend
ax4.legend(['given_df', 'strm_mask'])


"""
# Create a new figure and axis object
fig, ax4 = plt.subplots()

# Convert to array
arr = np.array(dataframe_polygons_polar['geometry'].astype(float).tolist())

# Plot the array using imshow()
ax4.imshow(arr)

plt.tight_layout()
#ax4.figure.savefig(path_fig+"margold-2014_polygons.png", dpi=800)

plt.show()
plt.close()
"""
=======
>>>>>>> f3c55661d006a57fb60de522a513d7fa008519ee
