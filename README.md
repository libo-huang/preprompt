# PrePrompt: Predictive Prompting for Class Incremental Learning

## Requirements
The experiments were conducted on an 8xNVIDIA RTX 4090 GPU cluster with the following environment:

```bash
conda create -n preprompt python=3.8
conda activate preprompt
pip install -r requirements.txt
```

Required packages (`requirements.txt`):
```
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
torchprofile==0.0.4
torch==1.13.1
torchvision==0.14.1
urllib3==2.0.3
scipy==1.7.3
scikit-learn==1.0.2
numpy==1.21.6
```

## Datasets
The following datasets are automatically downloaded and preprocessed when running the training scripts:
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz) 
- 5-Datasets (SVHN, MNIST, CIFAR-10, notMNIST, Fashion-MNIST)

## Reproduction
To reproduce our experiments, run the corresponding training scripts:

```bash
# For CIFAR-100
bash training_scripts/train_cifar100_vit.sh

# For ImageNet-R
bash training_scripts/train_imr_vit.sh

# For CUB-200 
bash training_scripts/train_cub_vit.sh

# For 5-Datasets
bash training_scripts/train_5datasets_vit.sh
```

## Acknowledgments
This implementation builds upon:
- [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch)
- [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)

We thank the original authors for sharing their code.