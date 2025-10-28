# PrePrompt: Predictive Prompting for Class-Incremental Learning

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2505.08586)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)


Official PyTorch implementation of **PrePrompt**, a novel predictive prompting framework for class-incremental learning (CIL). PrePrompt introduces a predictive mechanism that anticipates future task requirements, enabling more effective and efficient knowledge retention across sequential learning tasks.

> 📄 **Citation**: If you find our work useful, please consider citing:
> ```bibtex
> @article{huang2025preprompt,
>   title={PrePrompt: Predictive Prompting for Class Incremental Learning},
>   author={Huang, Libo and An, Zhulin and Yang, Chuanguang, et al},
>   journal={arXiv preprint arXiv:2505.08586},
>   year={2025}
> }
> ```

## 🚀 Key Features

- **Predictive Prompting**: Anticipates future task distributions to optimize prompt selection
- **State-of-the-Art Performance**: Achieves superior results on multiple CIL benchmarks
- **Efficient Adaptation**: Minimal computational overhead with maximal knowledge retention
- **Easy Integration**: Compatible with existing prompt-based CIL methods

## 📊 Performance Highlights

PrePrompt achieves state-of-the-art performance across multiple challenging class-incremental learning benchmarks (10 tasks with equal number of classes  of CIFAR-100, ImageNet-R, CUB-200 while 5 tasks of 5-Datasets):

| Dataset | Final Accuracy (%) | Average Incremental Accuracy (%) | Forgetting Rate (%) |
|---------|-------------------|----------------------------------|---------------------|
| CIFAR-100 | 93.74 | 95.41 | 1.27 |
| ImageNet-R | 75.09 | 78.96 | 1.11 |
| CUB-200 | 88.27 | 88.29 | 1.81 |
| 5-Datasets | 94.54 | 95.78 | 0.21 |

*Detailed results available in our [paper](https://arxiv.org/abs/2505.08586).*

## 🛠️ Installation

### Environment Setup
```bash
# Create and activate conda environment
conda create -n preprompt python=3.8 -y
conda activate preprompt

# Install dependencies (retry if network issues occur)
pip install -r requirements.txt
```

### Dependencies (requirements.txt)
``` text
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


## 📁 Datasets
The framework automatically downloads and preprocesses the following benchmark datasets:
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz): 100-class image classification dataset
- [ImageNet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar): ImageNet variants with artistic renditions
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz) : Fine-grained bird species classification
- 5-Datasets: Composite benchmark (SVHN, MNIST, CIFAR-10, notMNIST, Fashion-MNIST)

💡 Note: Pre-download datasets to `./datasets/` for faster setup or unstable networks.

## 🎯 Quick Start
Execute the corresponding training scripts for each benchmark:
```bash
# CIFAR-100 Experiments
bash training_scripts/train_cifar100_vit.sh

# ImageNet-R Experiments  
bash training_scripts/train_imr_vit.sh

# CUB-200 Fine-grained Classification
bash training_scripts/train_cub_vit.sh

# 5-Datasets Sequential Learning
bash training_scripts/train_5datasets_vit.sh
```



## 🙏 Acknowledgments
This implementation builds upon several excellent open-source projects:
- [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch): Foundation for prompt-based continual learning
- [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt): Insights into hierarchical prompt design

We sincerely thank the original authors for sharing their code and inspiring this work.

## 📜 License
This project is released under the MIT License. See LICENSE file for details.

## 📧 Contact
For questions and discussions, please open an issue or contact the maintainers.


<div align="center">
⭐ Don't forget to star this repository if you find it helpful!
</div>