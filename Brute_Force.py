# -*- coding: utf-8 -*-
"""
Compute by brute force all the combinations
Variables are named in a similar way as in PSO_GA_Train.py

@author: Victor Pozzobon

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""
# Clear all --- START
import os
import sys

### --- Clear all --- ###
from IPython import get_ipython
get_ipython().magic('reset -sf') 
### --- --------- --- ###


### Loading modules ###
import time
startTime = time.time()
startTime_string = time.localtime()

import numpy as np
import multiprocessing as mp
import sys
from itertools import combinations
from helpFunctions import *
import warnings
warnings.filterwarnings('ignore')


##############
# Parameters #
##############
   
nbParam = 151     # Number of paramenters (= features, wavelength here) // PSO + GA
procNb = 8        # Number of tread available for parallel runs
maxObj = 1e300    # Unlikely to be met maximum of the cost function
    
# Get all combinations of 3 wavelength
comb = list(combinations(np.arange(0,151,1), 3))
nbRun = len(comb) # Run of combination to test

# Allocating mémory
m = np.zeros((nbParam + 1 + 1, nbRun + 1))

for i in range(0, nbRun):
    m[comb[i], i] = 1
    m[-1, i] = i
    m[-2, i] = maxObj
m[-2, -1] = maxObj
    
################
# Data loading #
################
Data_Reduced = loadData(nbParam, filename="Nitrate_Spectra_train.npy")

###########################
# Starting to brute force #
###########################

# Preparing to store the results
results = []
def accumulateResults(currentResult):
    results.append(currentResult)
     
# Filling the pool
pool_BF = mp.Pool(processes=procNb)
m_split = np.array_split(m[:,:-1], procNb, axis=1)

for j in range(0, len(m_split)):
    # Cost function computation in parallel
    pool_BF.apply_async(processLargeLoader, args=(m_split[j], False, True, Data_Reduced), callback=accumulateResults)
pool_BF.close()
pool_BF.join()

print("Parallel running done")

# Organising the results
for j in range(0, len(results)):
    for k in range(0, len(results[j])):
        m[nbParam, int(results[j][k][1])] = results[j][k][0]
        
# Processing the results to identify the best run
for j in range(0, nbRun):
    # Checking for the best run
    if m[nbParam, j] < m[nbParam, nbRun]:
        m[:, nbRun] = m[:,j]
        
# Storing the results
np.savetxt("Brute_force_last.txt", m[:, nbRun])  # Best one isolated
np.save("Brute_force", m)                        # All the tested combinations
    
endTime = time.time()
print('Execution time : {:3.2f} s'.format((endTime - startTime)))