# TherML: closed loop materials discovery of thermoelectric using error correction learning

## Table of Contents

- [How to cite](#how-to-cite)
- [Installation](#how-to-install)
- [Data](#data)
- [Structure of the repository](#structure-of-the-repository)
- [Usage](#usage)
- [License](#license)

### How to install: 
- First install the prerequisite libraries:
  * PyTorch, ASE, Pymatgen, TPOT, Scikit-learn, Matminer
- Clone this library locally as: ```git clone https://github.com/hitarth64/therml```
- Optional installs (refer to [optionals](therml/optionals.md) for more information):
  * ROOST from https://github.com/CompRhys/roost
  * CrabNet from https://github.com/anthony-wang/CrabNet

### Data: 
- We use two sets of data in this study:
  * Thermoelectric property database from [MRL @ UCSB](http://www.mrl.ucsb.edu:8080/datamine/thermoelectric.jsp)
  * Closed-loop [experimental dataset](therml/featurized_experimental_data.pkl)

### Structure of the repository:
- ```therml``` directory:
  * Contains files, modules and data required for training accurate ML model to perform error-correction learning and rank materials
  * Please refer to the in-directory README file for more information
- ```therml/saved_models``` directory:
  * Contains the checkpoint of our model with highest cross-validation score 
- ```therml/prior_models``` directory:
  * All the prior-models trained using Magpie, Roost and CrabNet (refer to manuscript for definition of prior-models)
  * Please refer to the in-directory README file for more information

### Usage:

- You can perform the inference using:
  * ```python inference.py```
  * Modify the inquiry dataloader within ```inference.py``` to rank new material candidates
  
- You can perform error-correction learning using:
  * ```python hpo_dense.py```
  * Enables you to perform hyperparameter search for the error-correction model.
  * It is setup by default, to train and cross-validate on all the data collected until the last round (which is what we did)
  
- If you encounter any problem, feel free to start a discussion in the **Issues**

### How to cite:
```
@article{https://doi.org/10.1002/adma.202302575,
author = {Choubisa, Hitarth and Haque, Md Azimul and Zhu, Tong and Zeng, Lewei and Vafaie, Maral and Baran, Derya and Sargent, Edward H},
title = {Closed-loop Error Correction Learning Accelerates Experimental Discovery of Thermoelectric Materials},
journal = {Advanced Materials},
volume = {n/a},
number = {n/a},
pages = {2302575},
doi = {https://doi.org/10.1002/adma.202302575},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/adma.202302575},
 ```

### License:
TherML is released under the MIT License. 
