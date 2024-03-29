# Particle Swarm & Genetic Algorithm Optimizer associated with Partial Least Square Regression for nitrate and nitrite spectroscopic quantification 

Repository hosting the Python codes (please be kind with the typos in the comments) and data associated with the article: 
[Nitrate and nitrite as mixed source of nitrogen for _Chlorella vulgaris_: fast nitrogen quantification using spectrophotometer and machine learning.](https://link.springer.com/article/10.1007/s10811-021-02422-2) 
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
*Journal of Applied Phycology*, 1-9. [(Preprint)](https://victorpozzobon.github.io/assets/preprints/Pozzobon_2021_b.pdf)

It has been tested successfully on July 2021.

## Data structure

The whole curated dataset is contained within _Nitrate_Spectra.npy_ (261 measurements). It was randomly split into _Nitrate_Spectra_train.npy_ containing 80 % of the data and _Nitrate_Spectra_valid.npy_ hosting the remaining part. Individual measurements are stored as columns. The 151 first items are the spectrophotometric readings (from 190 to 340 nm), and the two remaining rows are nitrite (item 152) and nitrate (item 153) concentrations. Files are available in the _Data_ folder.

## How to run

__Step 1: determine the most relevant wavelengths__

Run _PSO_GA_Train.py_ (calling _helpFunctions.py_) to run the PSO GA hybrid 1000 times. For each run, it will determine a set of wavelengths allowing to describe the data of the _Nitrate_Spectra_train.npy_ dataset. All the runs are stored in the _Individual_Runs_ folder. For testing purposes, mine are stored in the archive _Run\_n.tar.xz_ in the _Individual_Runs_ folder.

__Step 2: sort the wavelengths by occurence__

Run _Agglomerate_Individual_Run.py_ to sort the wavelengths by occurrence (it will plot a top 10 by default).

__Step 3: determine the adequate number of components__

By including wavelengths successively, _ScreePlot.py_ produces a scree plot, allowing to determine the relative improvement associated with the addition of an extra wavelength. The PRESS metric is used on the _Nitrate_Spectra_train.npy_ dataset. Choosing three wavelengths is the optimum in this case. 

__Step 4: brute force__

_BruteForce.py_ tests all the possible combinations of three wavelengths. At this point, we should have a good idea of which wavelengths are going to win. The objective is to confirm it. For this, we are still using the training dataset (_Nitrate_Spectra_train.npy_). 

__Step 5: validation__

The produced equations are finally tested on the validation dataset (_Nitrate_Spectra_valid.npy_) and plotted. To do so, run _Validation.py_. 

All the results files are stored within the _Results_ folder. Here is a picture of the final validation. 

![Image not found](./Results/Results.png?raw=true)

__Comment__

After some additional work on an unrelated project, I realized that my implementation of the PSO GA hybrid was not identical to the one in the article I am referring to. While it is undoubtedly suboptimal compared to Gong's proposal, it is functional. Therefore, for the sake of reproducibility, I decided to leave the flawed version online. Still, I shall publish the properly implemented version in another repository associated with another work requiring an efficient optimization procedure.

## Contact

Please feel free to contact me. You can find my details on: https://victorpozzobon.github.io/


