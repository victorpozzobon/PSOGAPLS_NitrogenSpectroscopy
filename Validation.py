# -*- coding: utf-8 -*-
"""
Validation, plotting results and getting the equations

@author: Victor Pozzobon

Article
Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.
https://www.springer.com/journal/10811
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
Journal of Applied Phycology, 1-9.
"""
#### --- Clear all --- ###
Clear = False
if Clear:
    from IPython import get_ipython
    get_ipython().magic('reset -sf') 
#### --- --------- --- ###

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ShuffleSplit
import numpy as np


# Loading the best set of wavelengths
m = np.genfromtxt("Results/Brute_force_last.txt")

# Training PLS
Data_train = np.load("Data/Nitrate_Spectra_train.npy")
X = Data_train[:-2, :][m[:-2]>0.5]
Y = Data_train[-2:, :]

pls = PLSRegression(n_components = 3)
pls.fit(np.transpose(X), np.transpose(Y))

# Validating
Data_valid = np.load("Data/Nitrate_Spectra_valid.npy")
X = Data_valid[:-2, :][m[:-2]>0.5]
Y = Data_valid[-2:, :]
Y_pred = pls.predict(np.transpose(X))
Y = np.transpose(Y)

# Saving data for reuse or plot in another software
np.savetxt('Results/Data_validation.txt', Y)
np.savetxt('Results/Data_prediction.txt', Y_pred)

########################
# Plotting the results #
########################
nrow = 3
ncol = 2
plt.figure(figsize=(12, 12)) 


# First bissector
# NO2
plt.subplot(nrow, ncol, 1)

# Bounding the graphs
mingraph = 0.9 * np.min([np.min(Y_pred[:,0]), np.min(Y[:,0])])
maxgraph = 1.1 * np.max([np.max(Y_pred[:,0]), np.max(Y[:,0])])
plt.xlim([mingraph, maxgraph])
plt.ylim([mingraph, maxgraph])

mingraph = 0
plt.plot([mingraph, maxgraph], [mingraph, maxgraph], '-b')
plt.legend(["First bisector"])
plt.title("First bisector plot NO2")
plt.xlabel("PSO-GA-PLS estimations (mg/l)")
plt.ylabel("IC measurements (mg/l)")

# Labeling with the numbers
number = True
if number:
    for k in range(0, len(Y_pred[:,0])):
        label = str(k)
        plt.annotate(label, # this is the text
                     (Y_pred[k,0],Y[k,0]), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
plt.plot(Y_pred[:,0], Y[:,0], 'x', color = 'r')

# NO3
plt.subplot(nrow, ncol, 2)

# Bounding the graphs
mingraph = 0.9 * np.min([np.min(Y_pred[:,1]), np.min(Y[:,1])])
maxgraph = 1.1 * np.max([np.max(Y_pred[:,1]), np.max(Y[:,1])])
plt.xlim([mingraph, maxgraph])
plt.ylim([mingraph, maxgraph])
mingraph = 0

plt.plot([mingraph, maxgraph], [mingraph, maxgraph], '-b')
plt.legend(["First bisector"])
plt.title("First bisector plot NO3")
plt.xlabel("PSO-GA-PLS estimations (mg/l)")
plt.ylabel("IC measurements (mg/l)")

# Labeling with the numbers
if number:
    for k in range(0, len(Y_pred[:,1])):
        label = str(k)
        plt.annotate(label, # this is the text
                     (Y_pred[k,1],Y[k,1]), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,0), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
plt.plot(Y_pred[:,1], Y[:,1], 'x', color = 'r')

# Absolute comparison
# NO2 
plt.subplot(nrow, ncol, 3)
plt.plot(Y[:,0], 'o')
plt.plot(Y_pred[:,0], 'x', color = 'r')
plt.legend(["IC measurements", "PSO-GA-PLS estimation"])
plt.title("Direct comparison - On the validation part of the dataset - NO2")
plt.xlabel("Index (-)")
plt.ylabel("Concentration (mg/l)")

# NO3
plt.subplot(nrow, ncol, 4)
plt.plot(Y[:,1], 'o')
plt.plot(Y_pred[:,1], 'x', color = 'r')
plt.legend(["IC measurements", "PSO-GA-PLS estimation"])
plt.title("Direct comparison - On the validation part of the dataset - NO3")
plt.xlabel("Index (-)")
plt.ylabel("Concentration (mg/l)")

# Error plot
# NO2 
plt.subplot(nrow, ncol, 5)
n, bins, patches = plt.hist(x=Y_pred[:,0] - Y[:,0], bins='auto', color='#0504aa',
                        alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Error value (mg/l)')
plt.ylabel('Frequency (%)')

# NO3
plt.subplot(nrow, ncol, 6)
n, bins, patches = plt.hist(x=Y_pred[:,1] - Y[:,1], bins='auto', color='#0504aa',
                        alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Error value (mg/l)')
plt.ylabel('Frequency (%)')

# Saving the graphs
plt.tight_layout()    
plt.savefig("Results/Results.png", dpi= 600)

#########################
# Computing some values #
#########################
print("Various value assessing quantification efficiency:")
print("Mean NO2 error: {:3.3f} mg/l".format(np.average(Y_pred[:,0] - Y[:,0])))
print("Mean NO3 error: {:3.3f} mg/l".format(np.average(Y_pred[:,1] - Y[:,1])))
print("STD NO2 error: {:3.3f} mg/l".format(np.std(Y_pred[:,0] - Y[:,0])))
print("STD NO3 error: {:3.3f} mg/l".format(np.std(Y_pred[:,1] - Y[:,1])))

# Relative errors
err_perc_NO2 = 0
k = 0
for i in range(0, len(Y[:,0])):
    if Y[i,0] > 0:
        err_perc_NO2 += (Y_pred[i,0] - Y[i,0]) / Y[i,0] * 100
        k +=1
print("Abs % NO2 error: {:3.3f} %".format(err_perc_NO2/k))

err_perc_NO3 = 0
k = 0
for i in range(0, len(Y[:,1])):
    if Y[i,1] > 0:
        err_perc_NO3 += (Y_pred[i,1] - Y[i,1]) / Y[i,1] * 100
        k +=1
print("Abs % NO2 error: {:3.3f} %".format(err_perc_NO3/k))

# Removing outliers (should not impact results)
mask= np.abs(Y_pred[:,0] - Y[:,0]) < 0.5
print("STD NO2 error without outlier: {:3.3f} mg/l".format(np.std(Y_pred[mask,0] - Y[mask,0])))
mask= np.abs(Y_pred[:,1] - Y[:,1]) < 0.5
print("STD NO3 error without outlier: {:3.3f} mg/l".format(np.std(Y_pred[mask,1] - Y[mask,1])))
        
#LOD and LOQ
# NO2
NO2_blank = []
for i in range(0, len(Y[:,0])):
    if Y[i,0] < 0.1:
        NO2_blank.append(Y_pred[i,0])
        
print("NO2_Blank avg: {:3.3f} mg/l".format(np.average(NO2_blank)))
print("NO2_Blank std: {:3.3f} mg/l".format(np.std(NO2_blank)))
print("NO2 LoD: {:3.3f} mg/l".format(np.average(NO2_blank)*0 + 3 * np.std(NO2_blank)))
print("NO2 LoQ: {:3.3f} mg/l".format(np.average(NO2_blank)*0 + 10 * np.std(NO2_blank)))

# NO3
NO3_blank = []
for i in range(0, len(Y[:,0])):
    if Y[i,1] < 0.1:
        NO3_blank.append(Y_pred[i,1])
        
print("NO3_Blank avg: {:3.3f} mg/l".format(np.average(NO3_blank)))
print("NO3_Blank std: {:3.3f} mg/l".format(np.std(NO3_blank)))
print("NO3 LoD: {:3.3f} mg/l".format(np.average(NO3_blank)*0 + 3 * np.std(NO3_blank)))
print("NO3 LoQ: {:3.3f} mg/l".format(np.average(NO3_blank)*0 + 10 * np.std(NO3_blank)))


##################################
# Obtaining the actual equations #
##################################

# Using symbolic calculation to get an "esay to read" output
import sympy as sym
from sympy.physics.vector import ReferenceFrame, dot
from sympy.matrices import MatrixSymbol, Transpose
a1 = sym.Symbol('WL_1')
a2 = sym.Symbol('WL_2')
a3 = sym.Symbol('WL_3')

xsym = sym.Matrix([a1, a2, a3])
coef = sym.Matrix(pls.coef_)
xmean = sym.Matrix(pls.x_mean_)
xstd = sym.Matrix(pls.x_std_)
ymean = sym.Matrix(pls.y_mean_)

X1 = xsym-xmean
for i in range(0, len(X1)):
    X1[i] = X1[i] / xstd[i]

expression =Transpose(X1)*coef+Transpose(ymean)

print("\nSymbolic:")
print('Equation, then coefficients for each species with adequate precision')
k = 0
for chemical in ["NO2", "NO3"]:
    print(chemical)
    print(expression[k])
    for i in range(0, len(xsym)):
        print("For " + str(xsym[i]) + ': {:3.3f}'.format(expression[k].coeff(xsym[i])))
    k = k + 1    

