
"""
Created on Sun Jun 28 00:40:40 2020

@author: shahadat hossain, 3847363
"""

import numpy as np
import xarray as xr
#import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec


path = '/home/shihsir/Desktop/Codes/Trachte/'
file = 'airUV.mon_mean.1981-2010.EU.nc' 
srcXR = xr.open_dataset(path+file)
srcXR.dims
srcXR.info()
srcXR.T.dims
srcXR.T.values
srcXR.lon.values

#Calculating mean values over seasons

srcXR_season = srcXR.groupby('time.season').mean('time')
srcXR_season.season.values
print(srcXR_season)
print(srcXR)
srcXR_season.T.dims
srcXR_season.lat.values
srcXR_season.lon.values
srcXR_season.T[0,:,:]

#Calcualring climatology over entire time

src_mean = srcXR.T.mean('time')
print(src_mean)
src_mean.dims


#Plotting climatology

selected_coordinate = [-10, 15, 50, 40]

fig, tem = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

tem.add_feature(cf.COASTLINE.with_scale('50m'),linewidth=0.2, zorder=4)
tem.add_feature(cf.BORDERS.with_scale('50m'),  linewidth=0.2, zorder=4)

tem_range = np.arange(-15., 35., 5.)
temp_contour = tem.contourf(src_mean.lon, src_mean.lat, src_mean,
                            levels= tem_range, cmap= get_cmap("rainbow"))
tem.set_extent([-10, 15, 50, 40])
tem_djf = plt.colorbar(temp_contour, ax=tem, 
                       fraction=0.05,
                      pad=0.03,shrink=0.6,
                      ticks=[-15, 0, 15, 30])
tem_djf.ax.tick_params(labelsize=8)
tem_djf.ax.set_title('C', pad=4.0, ha= 'center', fontsize= 8)

tem.set_xticks(np.arange(-5, 20, 5),crs=ccrs.PlateCarree())
tem.set_yticks(np.arange(40, 55, 5),crs=ccrs.PlateCarree())
tem.xaxis.set_major_formatter(LongitudeFormatter())
tem.yaxis.set_major_formatter(LatitudeFormatter())
tem.tick_params(reset=True,axis='both',which='major',
                labelsize=8,direction='in',
                bottom = True, top = True, 
                left = True, right = True, 
                width = 0.2, labelbottom=True, zorder=6) 
tem.outline_patch.set_linewidth(0.2)
tem.outline_patch.set_zorder(6)
#tem.set_xlim(srcXR.lon[0], srcXR.lon[-1])
#tem.set_ylim(srcXR.lat[-1],srcXR.lat[0])
tem.text(0, 1.1, 'Climtotogy over time', transform=tem.transAxes, fontsize=10, 
            fontweight='bold',color='navy', va='center', ha='left')    
    

#Plotting over seasons

dnTXT = {0: 'DJF', 1: 'JJA', 2:'MAM', 3:'SON'}
tem_range = np.arange(-15., 40., 5.)
srcXR_season.U[1,::, ::]

def plotRH(dn,tn):
    dn.add_feature(cf.COASTLINE.with_scale('50m'),linewidth=0.2, zorder=4)
    dn.add_feature(cf.BORDERS.with_scale('50m'),  linewidth=0.2, zorder=4)
    dn_rh_contours = dn.contourf(srcXR_season.lon,srcXR_season.lat,srcXR_season.T[tn,:,:],
                             levels=tem_range,cmap=get_cmap("rainbow"),
                             transform=ccrs.PlateCarree())
    
    cb_dn = plt.colorbar(dn_rh_contours, ax=dn,
                      fraction=0.05,
                      pad=0.03,shrink=0.9,
                      ticks=[-15, -5, 5, 15, 30])
    cb_dn.ax.tick_params(labelsize=8)
    cb_dn.ax.set_title('C',pad=4.0,ha='center',fontsize=8)
    cb_dn.ax.set_anchor('W')
    
    dn.set_xticks(np.arange(-5, 20, 5),crs=ccrs.PlateCarree())
    dn.set_yticks(np.arange(40, 55, 5),crs=ccrs.PlateCarree())
    dn.xaxis.set_major_formatter(LongitudeFormatter())
    dn.yaxis.set_major_formatter(LatitudeFormatter())
    dn.tick_params(reset=True,axis='both',which='major',
                labelsize=8,direction='in',
                bottom = True, top = True, 
                left = True, right = True, 
                width = 0.2, labelbottom=True, zorder=6) 
    dn.outline_patch.set_linewidth(0.2)
    dn.outline_patch.set_zorder(6)
    dn.set_xlim(srcXR.lon[0], srcXR.lon[-1])
    dn.set_ylim(srcXR.lat[-1],srcXR.lat[0])
    dn.text(0, 1.1, 'Season '+ dnTXT[tn], transform=dn.transAxes, fontsize=10, 
            fontweight='bold',color='blue', va='center', ha='left')    
    
 
    q = dn.quiver(srcXR_season.lon[::2],srcXR_season.lat[::2],
                          srcXR_season.U[tn,::2, ::2],srcXR_season.V[tn,::2, ::2],
                          color='red')
    dn.quiverkey(q, X=0.0, Y= -0.2, U=5,
             label='Quiver key, len = 5', labelpos='E',color='red')
    

# plotting   
    
fig = plt.figure() 
gs = fig.add_gridspec(2,2, height_ratios=[1,1], hspace=.65) 

for i in range(0,4):
    sfig = fig.add_subplot(gs[i],projection=ccrs.PlateCarree())
    plotRH(sfig,i)
