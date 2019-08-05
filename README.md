## Automatic Generation of Personalized Comment Based on User Profile
- This is the code for ACL 2019 SRW paper *[AGPC: Automatic Generation of Personalized Comment Based on User Profile](https://arxiv.org/pdf/1907.10371v1.pdf)* 

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
Please kindly cite our paper if this [paper](https://arxiv.org/pdf/1907.10371v1.pdf) or the code are helpful.
```
@article{zeng2019automatic,
  title={Automatic Generation of Personalized Comment Based on User Profile},
  author={Zeng, Wenhuan and Abuduweili, Abulikemu and Li, Lei and Yang, Pengcheng},
  journal={ACL 2019},
  pages={229},
  year={2019}
}
```
