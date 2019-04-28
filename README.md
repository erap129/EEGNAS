## EEGNAS: an EEG-targeted Neural architecture search algorithm

###Instructions to run example experiment on the BCI Competition IV 2a dataset:
1. git clone the project to your machine
2. from the main folder 'BCI_Benchmarks' run: `python BCI_IV_2a_experiment.py -e cross_per_subject`
3. This will run 2 experiment configurations:
    1. Cross subject directed NAS (initial data loading takes ~1 min)
    2. Per subject directed NAS
    
* Note: On a high-end Nvidia GPU each configuration should take 1-2 days to complete.
* For test purposes, it is possible to edit the 'config.ini' file, located in the 'configurations' folder.
    * For example, the property 'num_generations' can be changed from 75 to any other number, in order to get shorter run-times (and worse results).
    
    
### Additional Notes
* Datasets:
    * The BCI Competition IV 2a dataset is included in the repository. No further action required.
    * BCI Competition IV 2b dataset - taken from the [moabb](https://github.com/NeuroTechX/moabb) BCI toolbox.
    * High Gamma dataset - need to [download manually](https://web.gin.g-node.org/robintibor/high-gamma-dataset) and move to the folder 'data/HG'
    * Inria BCI dataset - need to [download from kaggle](https://www.kaggle.com/c/inria-bci-challenge/data) and move to the folder 'data/NER15'
    * Opportunity dataset - need to download manually and move to the folder 'data/Opportunity'. Use preproccesing code from [this repository](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs/blob/master/data/download_datasets.py) to obtain the file 'oppChallenge_gestures.data', which needs to be put in the folder 'data/Opportunity'

* EEGNAS is **work in progress**. In the future will be added an option to automatically analyze your own data and receive a neural architecture. Meanwhile, interested users can try to understand the NAS process themselves and add their own DB (the main task is to edit the file BCI_IV_2a_experiment.py and configurtions/config.ini)
* The repository is **heavy** because of the BCI Competition IV 2a dataset. It may take a while to download.