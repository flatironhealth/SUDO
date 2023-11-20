Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# SUDO
This repository contains the code required to conduct the experiments presented in the SUDO paper. 
Please note that the datasets to conduct such experiments are *not* provided here. 
If the datasets are publicly available, you can find links to them below. 

## Datasets
The experiments are conducted on several datasets:
1. Multi-Domain Sentiment (https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
2. Stanford DDI (https://ddi-dataset.github.io/index.html#access)
3. Camelyon17-WILDS (https://wilds.stanford.edu/get_started/)
4. Simulations

## Conducting experiments
To conduct the SUDO experiments, you need to follow two steps:
* Step 1 - Download the data of interest (from above links)
* Step 2 - Extract features (or prediction probabilities) from the data
* Step 3 - Perform SUDO experiments

### Step 2 - Feature extraction
To extract features from the datasets, please refer to the scripts entitled ```extract_XXX_features.py``` where XXX is a particular dataset's name

### Step 3 - SUDO experimentation
To perform SUDO experiments, please refer to the scripts entitled ```train_XXX_data.py```, where XXX is the particular dataset's name

#### Example
If you would like to conduct SUDO experiments on the Stanford DDI data, then you have to first run the code in ```extract_ddi_features.py``` and subsequently run the code in ```train_ddi_data.py```. At present, these scripts cannot be implemented from the command line. 


