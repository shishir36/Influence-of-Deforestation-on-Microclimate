"""
Shahadat Hossain
3847363

"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

path = '/home/shihsir/Desktop/Codes/Trachte/'
file = 'fnl.20180629.850_500hPa.EU.nc' 
srcXR = xr.open_dataset(path+file)

rh = srcXR['rh']
rh_500 = rh.sel(lev=50000)

##colud not slice data based on lon lat

rh_500_day = rh_500.sel(time='2018-06-29T12:00:00') 
rh_500_night = rh_500.sel(time='2018-06-29T00:00:00') 

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

fig, time = plt.subplots(2,1, subplot_kw={'projection': ccrs.PlateCarree()})

time[0].add_feature(cf.COASTLINE.with_scale('50m'),linewidth=0.2, zorder=4, edgecolor='k')
time[0].add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.2, zorder=4)
time[1].add_feature(cf.COASTLINE.with_scale('50m'),linewidth=0.2, zorder=4, edgecolor='k')
time[1].add_feature(cf.BORDERS.with_scale('50m'),linewidth=0.2, zorder=4)


rh_levels = np.arange(0., 101., 5.) 
day_rh_contours = time[0].contourf(rh_500_day.lon,rh_500_day.lat,rh_500_day,
                             levels=rh_levels,cmap=get_cmap("rainbow"),
                             transform=ccrs.PlateCarree())
night_rh_contours = time[1].contourf(rh_500_night.lon,rh_500_night.lat,rh_500_night,
                             levels=rh_levels,cmap=get_cmap("rainbow"),
                             transform=ccrs.PlateCarree())
plt.colorbar(night_rh_contours, ax=time,fraction=0.05,pad=0.03,shrink=0.9,ticks=[0,20,40,60,80,100])


time[0].set_xticks(np.arange(-15, 45, 5),crs=ccrs.PlateCarree())
time[0].set_yticks(np.arange(35, 65, 5),crs=ccrs.PlateCarree())

time[1].set_xticks(np.arange(-15, 45, 5),crs=ccrs.PlateCarree())
time[1].set_yticks(np.arange(35, 65, 5),crs=ccrs.PlateCarree())









