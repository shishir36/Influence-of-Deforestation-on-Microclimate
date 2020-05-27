# -*- coding: utf-8 -*-
"""
Created on Sun May  3 21:43:58 2020

@author: Shahadat Hossain
Mat No : 3847363
"""
import os
import numpy as np
import pandas as pd

#Exercise1
array= np.array([[1,2,3,4,5],[5,4,3,2,1],[0,9,8,7,6]], dtype=float)
df_empty= pd.DataFrame()
df_rw = pd.DataFrame(index=['rw1', 'rw2', 'rw3', 'rw4', 'rw5'])
df_rc = pd.DataFrame(columns=['clm1', 'clm2'], index=['rw1', 'rw2', 'rw3'])

#Exercise2
list=[1,2,3,4,5]
list.append('AA')
list.remove(2)
len(list)

#Exercise3
os.chdir('/media/shihsir/HDD/Courses/Microclimates')
Table = pd.read_csv('climate.dat', sep=";")
mean3= Table.iloc[:, 5:8].mean()


t = Table.loc[:,"T"].mean()
tmax = Table.loc[:,"Tmax"].mean()
tmin = Table.loc[:,'Tmin'].mean()
rr = Table.loc[:,"RR"].mean()
sun = Table.loc[:,"Sun"].mean()
meantbl = [t,tmax,tmin,rr,sun]
mean_df = pd.DataFrame(data=meantbl, columns=['Mean'], index= ['T','Tmax', 'Tmin', 'RR', 'Sun'])
print(mean_df)
