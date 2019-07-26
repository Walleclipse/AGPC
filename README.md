## Automatic Generation of Personalized Comment Based on User Profile
- This is the code for ACL 2019 SRW paper *[AGPC: Automatic Generation of Personalized Comment Based on User Profile](https://arxiv.org/abs/1907.10371)* 

### Requirements
* Python 3.5
* tensorflow 1.4

### Preprocessing
```
python prep_data.py 
```
We provide the sample data in sample_data/sample_data.csv

***************************************************************

### Training
```
python train_PCGN.py
```
All configurations and hyperparameters of the model are in configs/pcgn_config.yaml
****************************************************************

### Inference and Evaluation
```
python infer_PCGN.py
```
### Citing this work
Please kindly cite our paper if this [paper](https://arxiv.org/abs/1907.10371) or the code are helpful.
```
@inproceedings{Zeng2019AutomaticGO,
  title={Automatic Generation of Personalized Comment Based on User Profile},
  author={Wenhuan Zeng and Abulikemu Abuduweili and Lei Li and Pengcheng Yang},
  booktitle={ACL 2019},
  year={2019}
}
```
