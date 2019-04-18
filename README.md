# Welcome to EEGNAS
## an EEG-targeted Neural architecture search algorithm

###Instructions to run example experiment on the BCI Competition IV 2a dataset:
1. git clone the project to your machine
2. from the main folder 'BCI_Benchmarks' run: `python BCI_IV_2a_experiment.py -e cross_per_subject`
3. This will run 2 experiment configurations:
    1. Cross subject directed NAS
    2. Per subject directed NAS
    
* Note: On a high-end Nvidia GPU each configuration should take 1-2 days to complete.
* For test purposes, it is possible to edit the 'config.ini' file, located in the 'configurations' folder.
    * For example, the property 'num_generations' can be changed from 75 to any other number, in order to get shorter run-times (and worse results).