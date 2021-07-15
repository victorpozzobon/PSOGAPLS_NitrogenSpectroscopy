# Particle Swarm & Genetic Algorithm Optimizer associated with Partial Least Square for nitrate and nitrite spectroscopic quantification 

Repository hosting the Python codes and data associated with the article: 
[Nitrate and nitrite as mixed source of nitrogen for Chlorella vulgaris: fast nitrogen quantification using spectrophotometer and machine learning.](https://www.springer.com/journal/10811) 
Pozzobon, V., Levasseur, W., Guerin, C. & Perre, P. (2021). 
*Journal of Applied Phycology*, 1-9. [(Preprint)](https://victorpozzobon.github.io/assets/preprints/Pozzobon_2021_b.pdf)

It has been tested sucessfully on July 2021.

WORK IN PROGRESS !!!!!!!

## How to run

_Step 1: determine the most relevant wavelengths_
Run _PSO_GA_Train.py_ (calling _helpFunctions.py_) to run the PSO GA hybrid 2000 times. For each run, it will determine a set of wavelengths allowing to describe the data of the _Nitrate_Sepctra_train.npy_ dataset. 

_Step 2: sort the wavelengths by occurence_
Run _Agglomerate_Individual_Run.py_ to sort the wavelengths by occurrence (it will plot a top 10 by default).

_Step 3: determine the adequate number of components_
By including wavelengths successively, _ScreePlot.py_ produces a scree plot, allowing to determine the relative improvement associated with the addition of an extra wavelength. The PRESS metric is used on the _Nitrate_Sepctra_train.npy_ dataset. Choosing three wavelengths is the optimum in this case. 

_Step 4: brute force_
_BruteForce.py_ tests all the possible combinations of three wavelengths. At this point, we should have a good idea of which wavelengths are going to win. The objective is to confirm it. For this, we are still using the training dataset (_Nitrate_Sepctra_train.npy_). 

_Step 5: validation_
The produced equations are finally tested on the validation dataset (_Nitrate_Sepctra_valid.npy_) and plotted. To do so, run _Validation.py_. 

Here is a picture of the results
![Image not found](./Results.png?raw=true)

## Contact

Please feel free to contact me. You can find my details on: https://victorpozzobon.github.io/
