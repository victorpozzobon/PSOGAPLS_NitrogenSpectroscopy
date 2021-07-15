# PSOGAPLS_Nitrogen Spectroscopy

Repository hosting the OpenFOAM solver and tutorial associated with the article: 
[Numerical determination of the volumetric heat transfer coefficient in fixed beds of wood chips.](https://www.sciencedirect.com/science/article/abs/pii/S1385894720341140) 
Sassanis, V., Gamet, L., Rolland, M., Ma, R., &amp; Pozzobon, V. (2020). 
*Chemical Engineering Journal*, 128009.

It has been tested sucessfully on July 2021.

WORK IN PROGRESS !!!!!!!

## Structure

You can find the solver to compile in the *simpleThermFoam_solver* folder.
The associated tutorial can be found in *simpleThermFoam_tutorial/heatTransferSingleSphere*. It features a single sphere and the surrounding flow. The sphere exprience a temperature dependent heat realease that heats it up as well as the sweeping air.  
Please note that this example is a simplication of the case presented in the article. Its sole purpose is to run on a desktop (still in parallel) so that you can try it.

You can run the code with *Allrun* script and clean it with *Allclean*.

## Postprocessing

Here are two examples of vizualisations procuded with this tutorial.

Temperature field in the air and the sphere as well as streamlines (colored by velocity)
![Image not found](./Postprocessing_1.png?raw=true)

Inner sphere temperature field
![Image not found](./Postprocessing_2.png?raw=true)

## Contact

Please feel free to contact me. You can find my details on: https://victorpozzobon.github.io/
