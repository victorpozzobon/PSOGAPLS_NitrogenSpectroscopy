# -*- coding: utf-8 -*-
"""
PSO GA hybrid to screen which wavelengths appear the most often in
the PLS fit
Please see 
Gong Y, Li J, Zhou Y, Li Y, Chung HS, Shi Y, Zhang J (2016)
Genetic learning particle swarm optimization. IEEE Trans Cybern
46:2277–2290
for PSO GA hybrydation details

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
from helpFunctions import *
import warnings
warnings.filterwarnings('ignore')

##############
# Parameters #
##############

description = "PSO GA hybrid to screen which wavelengths appear the most often in the PLS fit"

# Tuning PSO + GA
nbParticle = 400                # Number of particles // PSO + GA
nbParam = 151                   # Number of paramenters (= features, wavelength here) // PSO + GA
plageUp   = 1*np.ones(nbParam)  # Upper bound of the range // PSO + GA
plageDown = 0*np.ones(nbParam)  # Lower bound of the range // PSO + GA
nbiter = 300                    # Maximum number of iteration  // PSO + GA
c1 = 0.6                        # Cognitive swarm parameter (0 to 1) // PSO 
c2 = 0.6                        # Social swarm parameter (0 to 1) // PSO 
inertie = 0.8                   # Intertia  // PSO overriden by chaotic one
pm = 0.01                       # Mutation probability // GA
speedThershold = 0.5            # For binary paramter, [0 1], value close to 1 restrict the chance of transition towards 1 // PSO
depMax = 5.0                        # Velocity capping // PSO - inactivated for binary parameter
                                    # ex: 100 => 1/100 of the range
plotPeriod = 1000               # Plot results every plotPeriod 
procNb = 8                      # Number of tread available for parallel runs
maxStagnation = 50              # Number of iretation which stagnate before stopping the search // PSO + GA
particleMaxStagnation = 15      # Number maxiumum of stagnation for a given particle, above go to the tournament // GA
tournamentSize = 0.2            # ]0,1] fracion of the swarm to sent to the tournament // GA

# Some verification and declaration
if min(plageUp - plageDown) < 0:
    print ("Check ranges!")
    sys.exit()
    
maxObj = 1e300                   # Unlikely to be met maximum of the cost function
Improvement = maxObj             # Storage to compute improvement from one iteration to the other
OldMax = maxObj                  # Former best results
StagnationCounter = 0            # Counter for stagnation

##################
# Initialization #
##################

### PSO Initialization ###
# Columns of m contains the p paramater and the cost function value f(p) in row p+1
m = np.ones((nbParam + 1 + 1, nbParticle + 1))       # each colums is a particle
                                                     # the last columns is the swarm best
                                                     # de toutes les particules et itération
                                                     # line nbParam+1 = best value
                                                     # line nbParam+2 = unique ID
                                                     
pbest = maxObj * np.ones((nbParam + 1, nbParticle))  # save the location associated to each particle best result
                                                     # and the associated cost function value
# Initializing velocity
speed = np.ones((nbParam, nbParticle))               # Velocity of each particle
for i in range(0, nbParam):
    speed[i,:] = (plageUp - plageDown)[i] / depMax * np.random.rand()
    
m[nbParam, :] = maxObj                               # Initializinf cost function
    
### GA Initialization  ###
o = np.zeros((nbParam+2+1, nbParticle)) # no population best, so nb_columns = nbParticle
                                        # line 1 to nbParam = parameter value
                                        # line nbParam +1 = best value
                                        # line nbParam +2 = unique ID
                                        # line nbPrama +3 = personnal stagnation
                                        
# Unique ID affectation   
for i in range(0, nbParticle):
    m[nbParam+1, i] = i 
    o[nbParam+1, i] = i 
        
################
# Data loading #
################

Data_Reduced = loadData(nbParam, filename="Data/Nitrate_Spectra_train.npy")

######################
# Starting main loop #
######################
# This loop will repeat the individual fitting procedure a large number of time
# you may not need to run it 1000 times, instead, you should monitor convergence
# of the retained wavelengths
nrunTot = 1000

for nrun in range(0, nrunTot):  # <= if need be you can restart from here by changing '0' to the last value
    # Re-initialization at the beginning of each search
    StagnationCounter = 0
    m[nbParam, :] = maxObj 
    
    # Sepcial initialization to prevent ecxessive number of wavelength
    for i in range(0, nbParticle):
        toOne = np.random.randint(0, nbParam)
        m[:nbParam+1, i] = 0
        m[toOne, i] = 1 # random sampling
    for i in range(0, nbParam):
        pbest[i, 0:nbParticle] = m[i, 0:nbParticle]
    
    # Before the first run, compute the starting point performances
    results = []
    def accumulateResults(currentResult):
        results.append(currentResult)
        
    # Filling the pool
    pool_PSO = mp.Pool(processes=procNb)
    
    # Splitting m over the different processors
    m_split = np.array_split(m[:,:-1], procNb, axis=1)
    
    for j in range(0, len(m_split)):
        # Cost function computation in parallel
        pool_PSO.apply_async(processLargeLoader, args=(m_split[j], False, True, Data_Reduced), callback=accumulateResults)
    pool_PSO.close()
    pool_PSO.join()

    # Organising the results
    for j in range(0, len(results)):
        for k in range(0, len(results[j])):
            m[nbParam, int(results[j][k][1])] = results[j][k][0]
            
    # Processing the results  
    for j in range(0, nbParticle):
        # Saving if beats particle best
        if m[nbParam, j] < pbest[nbParam, j]:
            pbest[:,j] = m[:-1,j]
        # Saving if beats swarm best
        if m[nbParam, j] < m[nbParam, nbParticle]:
            m[:, nbParticle] = m[:,j]
            

    #####################
    # Starting PSO + GA #
    #####################
    for i in range(0, nbiter):
        
        # Re-initialization at the beginning of each inner loop of the PSO GA
        nmutation = 0
        nswitch = 0
        # Define an output queue
        # Building up results agglomerator
        results = []
        def accumulateResults(currentResult):
            results.append(currentResult)
        
        
        ###               ###
        #    Genetic algo   #
        ###               ###
        
        # Cross over and mutation
        for j in range(0, nbParticle):
            for k in range(0, nbParam):
                ###             ###
                #    Cross over   #
                ###             ###
                # Randomly select annother particle
                prand = int(np.floor(np.random.uniform(0, nbParticle)))
                # if random particle best is better, cross over with it
                if m[nbParam, j] >= m[nbParam, prand]:       
                    z1 = np.random.uniform(0, 1)
                    o[k, j] = z1 * m[k, j] + (1-z1) * m[k, prand]
                # else keep current parameter value
                else:
                    o[k, j] = m[k, j]
                
                ###           ###
                #    Mutation   #
                ###           ###
                # just random
                if np.random.uniform(0, 1) < pm:
                    o[k, j] = int(abs(m[k,j] - 1))
                    nmutation += 1
                    
        # Is off spring better ?
        results = []
        # Filling the pool
        pool_GA = mp.Pool(processes=procNb)
        o_split = np.array_split(o[:-1,:], procNb, axis=1)
        
        for j in range(0, len(o_split)):
            # Cost function computation in parallel
            pool_GA.apply_async(processLargeLoader, args=(o_split[j], False, True, Data_Reduced), callback=accumulateResults)
        pool_GA.close()
        pool_GA.join()
    
        # Organising the results
        for j in range(0, len(results)):
            for k in range(0, len(results[j])):
                o[nbParam, int(results[j][k][1])] = results[j][k][0]
            
            
        # Counting paticle own stagnation and improving exemplar, if need be
        #    => first check if offspring better
        for j in range(0, nbParticle):
            # Not better
            if o[nbParam, j] > pbest[nbParam, j]:
                o[nbParam + 2, j] += 1
                
                # Tournament for too long stagnation
                if o[nbParam + 2, j] > particleMaxStagnation:
                    figthers_index = np.random.choice(nbParticle, int(np.floor(tournamentSize * nbParticle)), replace=False)
                    index_of_winner = np.where(pbest[nbParam, figthers_index] == np.amin(pbest[nbParam, figthers_index]))[0][0]
                    # Take the best parameters of the winner as exemplar
                    pbest[:,j] = pbest[:,index_of_winner]
                    
            # Offspring better, take it as new exemplar
            else:
                o[nbParam + 2, j] = 0
                pbest[:,j] = o[:-2,j]
    
        ###                   ###
        #    Part. Swarm Opt.   #
        ###                   ###     
        # Moving PSO particle accounting for offspring  
        for j in range(0, nbParticle):
            
            # Computing chaotic random inertia
            z1 = np.random.uniform(0, 1)
            z2 = np.random.uniform(0, 1)
            inertie = 0.5 * z1 + 4 * 0.5 * z2 * (1-z2)
            
            # Moving each particle
            for k in range(0, nbParam):
                # Cacul du déplacement
                z1 = np.random.uniform(0, 1)
                z2 = np.random.uniform(0, 1)
                targettedExemplar_k = (c1 * z1 * pbest[k, j] + c2 * z2 * m[k, nbParticle]) / (c1 * z1 + c2 * z2)
                
                speed[k, j] = inertie * speed[k, j] + np.random.rand() * (targettedExemplar_k - m[k,j])                    
                speedPrime = 1 / (1 + np.exp( - speed[k, j]))
                speedPrime = np.min([1, speedPrime])
                    
                # Updating each parameter
                if (speedPrime > np.random.uniform(speedThershold, 1)):
                    m[k,j] = int(abs(m[k,j] - 1))
                    nswitch += 1
                
                # Checking that the parameters are within the range
                if m[k, j] > plageUp[k]:
                    m[k, j] = plageUp[k]
                if m[k, j] < plageDown[k]:
                    m[k, j] = plageDown[k]
                    
        results = []
        # Filling the pool
        pool_PSO = mp.Pool(processes=procNb)
        m_split = np.array_split(m[:,:-1], procNb, axis=1)
        
        for j in range(0, len(m_split)):
            # Cost function computation in parallel
            pool_PSO.apply_async(processLargeLoader, args=(m_split[j], False, True, Data_Reduced), callback=accumulateResults)
        pool_PSO.close()
        pool_PSO.join()
    
        # Organising the results
        for j in range(0, len(results)):
            for k in range(0, len(results[j])):
                m[nbParam, int(results[j][k][1])] = results[j][k][0]
                
        # Processing the results  
        for j in range(0, nbParticle):
            # Saving if beats particle best
            if m[nbParam, j] < pbest[nbParam, j]:
                pbest[:,j] = m[:-1,j]
            # Saving if beats swarm best
            if m[nbParam, j] < m[nbParam, nbParticle]:
                m[:, nbParticle] = m[:,j]
    
    
    
        # Printing out
        print('\nIteration completed: ' + str(i+1))
        
        # Carrying on or stopping here?
        Improvement = np.floor(abs(OldMax - m[nbParam, nbParticle]) / (abs(OldMax) + 1e-15) * 100000)
        OldMax = m[nbParam, nbParticle]
        print('Improvement: ' + str(Improvement/1000) + ' %')
        print('Best score : {:f}'.format(m[nbParam, nbParticle]))
        print('Number of WL for best: ' + str(int(sum(m[:-2, nbParticle]))))
        print('Number of swicht : ' + str(nswitch))
        print('Number of mutation : ' + str(nmutation))

    
        if Improvement <= 0:
            StagnationCounter = StagnationCounter + 1
        else:
            StagnationCounter = 0
    
        print("Stagnation: " + str(StagnationCounter))
           
        if StagnationCounter > maxStagnation:
            break

        endTime = time.time()
        print('Execution time : {:3.2f} h to process: '.format((endTime - startTime)/3600) + str(nrun+1) + ' / ' + str(nrunTot))
        print('ETA: {:3.2f} h '.format((nrunTot-nrun-1) * (endTime - startTime) / (nrun+1)/3600))
        
        np.save("Individual_Runs/Run_" + str(nrun), m[:, nbParticle])
    
print('Overall process completed')
