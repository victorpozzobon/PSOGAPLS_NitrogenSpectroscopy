# -*- coding: utf-8 -*-
"""
PSO GA hybrid assistance functions
Load the data
Dispatch calculations
Compute PLS
@author: Victor Pozzobon

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""

'''
Loading the data
'''
def loadData(nbParam, filename = "Nitrate_Spectra.npy"):
    import numpy as np
    Data = np.load(filename)
    nfile = np.size(Data[0])
    nwl = np.size(Data[:,1])
    
    # Possible reduction of the data to increase speed
    # for workflow testing purpose, do not use in
    # production
    Data_Reduced = np.zeros((nbParam+2, nfile))
    width = int(151/nbParam)
    sampleWl = np.linspace(0, nwl -width -1, nbParam, dtype = int)
    
    for i in range(0, nfile):
        for j in range(0, nbParam):
            Data_Reduced[j, i] += np.sum(Data[j * width : (j+1) * width, i])/(width+1)

        Data_Reduced[-1, i] = Data[-1, i]
        Data_Reduced[-2, i] = Data[-2, i]
        
    return Data_Reduced

'''
Launching PLS of each particle of the batch
'''
def processLargeLoader(p, full = False, SupDataFlag = False, SupData = 0, flag_print = -1, startTime = 0):
    import numpy as np
    import time
    nb_particle_slip = p.shape[1]
    results = np.zeros([nb_particle_slip, 2])
    for i in range(0, nb_particle_slip):
        Gap = PLSFit(p[0:-2, i], full = False, SupDataFlag = False, SupData = SupData)
        results[i, :] = Gap, p[-1, i]
        if flag_print == 1:
            endTime = time.time()
            print('Execution time : {:3.2f} h to process: '.format((endTime - startTime)/3600) + str(i+1) + ' / ' + str(nb_particle_slip))
            print('ETA: {:3.2f} h '.format((nb_particle_slip-i-1) * (endTime - startTime) / (i+1)/3600))
        
    return results

'''
Computing the PLS results
'''
def PLSFit(p, full = False, SupDataFlag = False, SupData = 0, n_com = 5):

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import ShuffleSplit
    import numpy as np
    
    # Some variable
    pprime = np.zeros(len(p))
    pprime[:len(p)] = p
    pprime[-1] = 0
    
    # Training and validation sets
    train_index = []
    test_index = []
    fraction_valid = 0.2
    for i in range(0, SupData.shape[1]):
        # Avoid taking the 20 first percent to avoid bias
        if np.floor((fraction_valid * i) * 100) % 100 == 0:
            test_index.append(i)
        else:
            train_index.append(i)    
    
    K = float(sum(pprime>0.5)) # number of active parameter

    # Number of components for PLS, cannot be below number of paramter
    n_component_actual = np.min([n_com, int(K)])
    
    # You may want to run a convergence on the number of component,
    # in our case, 5 or more did not change de results

    # AIC metrci : 2 * k + n * log (RSS)
    # RSS for each cross validation set, hence, a sum
    Gap = 2 * K # => part 1 of AIC
    
    # Scaler
    calibration_factor = 1 - fraction_valid
    
    try:
        # Train and validate in cross validation
        X_calib = SupData[:-2, train_index][pprime>0.5]
        X_valid = SupData[:-2, test_index][pprime>0.5]
        Y_calib = SupData[-2:, train_index]
        Y_valid = SupData[-2:, test_index] 
        pls = PLSRegression(n_components = n_component_actual)
        pls.fit(np.transpose(X_calib), np.transpose(Y_calib))
        Y_pred = pls.predict(np.transpose(X_valid))
        Y_valid = np.transpose(Y_valid)
        Y_pred = (Y_pred + np.abs(Y_pred)) / 2 

        RSS = 0
        for i in range(0, len(Y_pred)):
            RSS += ((Y_pred[i,0] - Y_valid[i,0]))**2 + ((Y_pred[i,1] - Y_valid[i,1]))**2

        Gap += float(len(Y_valid)) / calibration_factor * np.log(RSS / calibration_factor) # => + n*log(RSS) | part 2 of AIC

    except Exception as e:
        Gap = 1e30
        # print("Crash") # Commented to silent the warning
        pass

    return Gap
