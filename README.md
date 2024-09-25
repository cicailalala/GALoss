# [ECCV2024] GALoss

This repository is the official implementation of **[Gradient-Aware for Class-Imbalanced Semi-supervised Medical Image Segmentation](https://eccv.ecva.net/virtual/2024/poster/1540), ECCV2024**. 

# Requirements:
- Python 3.7.11
- torch 1.10.0
- torchvision 0.11.0
- opencv-python 4.1.1.26
- numpy 1.20.3
- h5py 3.7.0
- scipy 1.7.1


# Datasets
**Dataset I**
[Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789). Following [DHC](https://github.com/xmed-lab/DHC), 20 samples were split for training, 4 samples for validation, and 6 samples for testing. We use the processed data by [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet).

**Dataset II**
[AMOS](https://amos22.grand-challenge.org/Dataset/). The processed dataset can be downloaded via [this link](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/qiwb_connect_hku_hk/Eq0j1GmOq-5AsRqPwTCgnrABjV3v-qYm4nZirzOiVN6ayw?e=OinMKI). Download and place the datasets in ```./data/```

**Dataset III**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/). We use the code and preprocessed data by [SSLMIS](https://github.com/HiLab-git/SSL4MIS/tree/master). 

# Running
```
CUDA_VISIBLE_DEVICES=0 python train_Synapse_CPS.py --seed 1337 --labelnum 4
```

## Reference
* [MagicNet](https://github.com/DeepMed-Lab-ECNU/MagicNet)
* [DHC](https://github.com/xmed-lab/DHC) 
## Citations

```bibtex
@inproceedings{qi2024gradient,
  title={Gradient-Aware for Class-Imbalanced Semi-supervised Medical Image Segmentation},
  author={Qi, Wenbo and Jiafei, Wu and Chan, SC},
  booktitle={ECCV2024},
  year={2024}
}
```
