# Automatic Generation of Personalized Comment Based on User Profile
- This is the code for ACL2019 student workshop paper *Automatic Generation of Personalized Comment Based on User Profile* 

## Requirements
* Python 3.6
* tensorflow 1.4

## Preprocessing
```
python prep_data.py 
```
We provide the sample data in sample_data/sample_data.csv

***************************************************************

## Training
```
python train_PCGN.py
```
All configuration and hyperparameter for the model in configs/pcgn_config.yaml
****************************************************************

## Inference and Evaluation
```
python infer_PCGN.py
```