# CNRMppe
Codebase for CNRM ensemble processing
## PPE files
 - CFMIP_Lambda.ipynb : Compute Lambda from different CFMIP outputs, different models and different simulations (amip-4XCO2, amip-pAK, amip-future4K).
## Processing files 
 - Preprocessing.ipynb : tests
 - Climatology_comparison.ipynb : tests
 - correction_SimulationsTXT.sh : Shell  script to find which simulations failed and did not give outputs in /scratch/globc/dcom/ARPEGE6_TUNE. Then save the rows number in a file called 'missing_lines_simulations.txt'
## Machine Learning files
 - NN_create.ipynb : 
     - Read 'simulations.csv' and create inputdata array : inputdata_file.npy (using 'missing_lines_simulations.txt' to delete the parameter dataset not used in simulations.)
     - Read '/PRE623TUN*.nc' and create outputdata array : outputdata_file.npy
     - Create and test out simple neural networks in Python with Keras. Based on Katie's code.
