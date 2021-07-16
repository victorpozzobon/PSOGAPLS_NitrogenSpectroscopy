# -*- coding: utf-8 -*-
"""
PRESS calculation and plotting

@author: Victor Pozzobon

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""

'''
Classical PRESS computation
'''
def PRESS(m, nbParticle, nbParam):
    import matplotlib.pyplot as plt
    import numpy as np
    
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import LeaveOneOut
    import numpy as np
    
    # Loading data
    Data_train = np.load("Data/Nitrate_Spectra_train.npy")
    X = Data_train[:-2, :][m[:-1,0]>0.5]
    Y = Data_train[-2:, :]
    
    K = float(sum(m[:-1,0]>0.5))
    n_component_actual = np.min([100, int(K)])
    Gap = 0
    bool_train = np.ones(X.shape[1], bool)
    bool_valid = bool_train.copy()
    
    # Leave one out sets
    for i in range(0, X.shape[1]):
        # Setting up index for training and validation
        bool_train[:] = True
        bool_train[i] = False
        bool_valid[:] = False
        bool_valid[i] = True
        
        X_calib = X[:, bool_train]
        X_valid = X[:, bool_valid]
        Y_calib = Y[:, bool_train]
        Y_valid = Y[:, bool_valid] 

        pls = PLSRegression(n_components = n_component_actual)
        pls.fit(np.transpose(X_calib), np.transpose(Y_calib))
        Y_pred = pls.predict(np.transpose(X_valid))
        Gap = Gap + (Y_valid[0,0] - Y_pred[0,0])**2 + (Y_valid[1,0] - Y_pred[0,1])**2 
        
    return Gap

import numpy as np
import time
startTime = time.time()
import matplotlib.pyplot as plt

# The wavelength sorted by occurence (highest first)
retainedWL = [6,44,34,11,10,9,56,45,16,46] # Attention: use indexes

# Number of feature / wavelength
nbParam = 151

# To store the restults
n_comp = []
MSE_Train = []

for i in range(0, len(retainedWL)):
    # "Turning on" the right wavelengths
    p = np.zeros((nbParam+1,2))
    p[retainedWL[:i+1],:] = 1
    n_comp.append(i+1)
    # Computing PRESS metric
    MSE_Train.append(PRESS(p, 0, nbParam))

# Plotting the results
plt.figure()
plt.semilogy(n_comp, MSE_Train, label="On the training dataset")
plt.xlabel('Number of components')
plt.ylabel('PRESS')
plt.legend()

np.savetxt("Results/PRESS_vs_Component.txt", np.transpose([n_comp, MSE_Train]), header="ncom TrainSetGap")

endTime = time.time()
print('\nExecution time : {:3.2f} s'.format((endTime - startTime)))