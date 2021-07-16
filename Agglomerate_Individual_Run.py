# -*- coding: utf-8 -*-
"""
Retrieve each run of the PSO GA algorithm 
Determine which wavelengths appear more often

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""

import os
import sys

### --- Clear all --- ###
from IPython import get_ipython
get_ipython().magic('reset -sf') 
### --- --------- --- ###

import time
startTime = time.time()
import numpy as np
import pylab
import matplotlib.pyplot as plt
plt.close('all')

# Where are the file stored
folder = "Individual_Runs"
nfile = 1000

# Getting the number of wavelengths
Data_0 = np.load(folder + "/Run_0.npy")[:-2]
nWL = len(Data_0)
Data = np.zeros((nWL, nfile)) # Allocating memory

# Loading all the files 
for i in range(0, nfile):
    Data[:,i] = np.load(folder + '/Run_' + str(i) + ".npy")[:-2]

# Computing frequencies
occurences = np.zeros([nWL, 2])
for i in range(0, nWL):
    occurences[i,0] = np.sum(Data[i, :]) / nfile * 100
    occurences[i,1] = i
occurences = occurences[occurences[:,0].argsort()] # Sorting them

# Ploting the top 10
top10string = "["    # To copy in the next routine
for i in range(0,10):
    print("Wavelength nb {:d}: {:d} nm occuring {:3.2f} % (index {:d})".format(i+1, 190 + int(occurences[-1-i, 1]), occurences[-1-i, 0], int(occurences[-1-i, 1])))
    top10string += str(int(occurences[-1-i, 1])) + ','

top10string = top10string[:-1] + ']'
print('\nTo avoid tedious manual report, maybe you can copy this string:')
print(top10string)

endTime = time.time()
print('\nExecution time : {:3.2f} s'.format((endTime - startTime)))