## Intro
This repository contains the source code used in the dissertation `Behavioural Correspondence of Neuronal Dynamics in 
Caenorhabditis elegans`.

## Datasets
The research dataset employed in this dessrtation, `Kato2015 whole brain imaging data`, was acquired from the OSF open repository. This invaluable resource, made accessible by `Manuel Zimmer`, can be accessed via the following URL: https://osf.io/2395t/?view_only=.

Opt for two datasets, namely, `WT_NoStim.mat` and `WT_Stim.mat`.

- `WT_NoStim dataset:` Signifying the wild type with the absence of sensory stimulation.

- `WT_Stim dataset:` Denoting the wild type wherein oxygen chemosensory neurons are activated via consecutive oxygen upshifts and downshifts (21% as opposed to 4%).

## Experimental Design
In this dissertation, two machine learning models, namely the `Logistic Regression Classifier` and the `Random Forest Classifier`, were chosen for analysis. These models were systematically trained and evaluated using two distinct datasets: `WT_NoStim` dataset and `WT_Stim` dataset.

- Train the `Logistic Regression Classifier` on the `WT_NoStim` dataset.
- Train the `Random Forest Classifier` on the `WT_NoStim` dataset.
- Train the `Logistic Regression Classifier` on the `WT_Stim` dataset.
- Train the `Random Forest Classifier` on the` WT_Stim` dataset.

## Jupyter Notebook
`Jupyter Notebook` used in this dissertation is stored in the directory `notebook`.

The notebook delineates the comprehensive workflow, encompassing stages of data preprocessing, model formulation and training, model validation, cross-validation procedures, hyperparameter optimization, among other processes.

## Package directory
Packages in this repository are available in the directory `src`.

- [model.py](src/model.py) Definition of `Logistic Regression Classifier` and `Random Forest Classifier` are established.
- [wt_nostim_data_preprocessing.py](src/wt_nostim_data_preprocessing.py) The `WT_NoStim` dataset undergoes preprocessing and is subsequently partitioned into training and test sets, adhering to a 7:3 ratio.
- [wt_nostim_train_validation.py](src/wt_nostim_train_validation.py) `Logistic Regression Classifier` and `Random Forest Classifier` are trained and evaluated utilizing the `WT_NoStim` dataset.
- [wt_stim_data_preprocessing.py](src/wt_stim_data_preprocessing.py) The `WT_Stim` dataset undergoes preprocessing and is subsequently partitioned into training and test sets, adhering to a 7:3 ratio.
- [wt_stim_train_validation.py](src/wt_stim_train_validation.py) `Logistic Regression Classifier` and `Random Forest Classifier` are trained and evaluated utilizing the `WT_Stim` dataset.
- [cross_validation.py](src/cross_validation.py) Average cross-validation scores for both the `Logistic Regression Classifier` and `Random Forest Classifier` are computed, considering both the entire `WT_NoStim` dataset and the entire `WT_Stim` dataset.
- [feature_importance.py](src/feature_importance.py) Some functions are established for quantifying the significance of neuronal features.
