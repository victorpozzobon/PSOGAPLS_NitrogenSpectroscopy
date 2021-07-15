# Particle Swarm & Genetic Algorithm Optimizer associated with Partial Least Square for nitrate and nitrite spectroscopic quantification 

Repository hosting the Python codes and data associated with the article: 
[Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.](https://www.springer.com/journal/10811) 
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
*Journal of Applied Phycology*, 1-9. [(Preprint)](https://victorpozzobon.github.io/assets/preprints/Pozzobon_2021_b.pdf)

It has been tested sucessfully on July 2021.

## Data structure

The whole curated dataset is contained within _Nitrate_Spectra.npy_ (261 measurements). It was randomly split into _Nitrate_Spectra_train.npy_ containing 80 % of the data and _Nitrate_Spectra_valid.npy_ hosting the remaining part. Individual measurements are stored as columns. The 151 first items are the spectrophotometric readings (from 190 to 340 nm), and the two remaining rows are nitrite (152) and nitrate (152) concentrations. 

## How to run

__Step 1: determine the most relevant wavelengths__

Run _PSO_GA_Train.py_ (calling _helpFunctions.py_) to run the PSO GA hybrid 1000 times. For each run, it will determine a set of wavelengths allowing to describe the data of the _Nitrate_Spectra_train.npy_ dataset. All the runs are stored in the _Individual_Runs_ folder

__Step 2: sort the wavelengths by occurence__

Run _Agglomerate_Individual_Run.py_ to sort the wavelengths by occurrence (it will plot a top 10 by default).

__Step 3: determine the adequate number of components__

By including wavelengths successively, _ScreePlot.py_ produces a scree plot, allowing to determine the relative improvement associated with the addition of an extra wavelength. The PRESS metric is used on the _Nitrate_Spectra_train.npy_ dataset. Choosing three wavelengths is the optimum in this case. 

__Step 4: brute force__

_BruteForce.py_ tests all the possible combinations of three wavelengths. At this point, we should have a good idea of which wavelengths are going to win. The objective is to confirm it. For this, we are still using the training dataset (_Nitrate_Spectra_train.npy_). 

__Step 5: validation__

The produced equations are finally tested on the validation dataset (_Nitrate_Spectra_valid.npy_) and plotted. To do so, run _Validation.py_. 

Here is a picture of the results
![Image not found](./Results.png?raw=true)

## Contact

Please feel free to contact me. You can find my details on: https://victorpozzobon.github.io/
