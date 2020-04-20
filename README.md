# Reproduction of BC learning for images
This repository is a reproduction study of [Between-class Learning for Image Classification](https://arxiv.org/abs/1711.10284) by Yuji Tokozume, Yoshitaka Ushiku, and Tatsuya Harada. 

This reproduction was made by Rickard Karlsson, Qian Bai and Erik Handberg. Our project and our results are presented in detail in this [blog post](https://bbbaiqian.github.io/CS4240_project).

## Between-class (BC) learning
Below is how the authors of the original study describe BC learning in their repository:

> #### Between-class (BC) learning:
> - We generate between-class examples by mixing two training examples belonging to different classes with a random ratio.
> - We then input the mixed data to the model and
train the model to output the mixing ratio.
> - Original paper: [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/abs/1711.10282) by us ([github](https://github.com/mil-tokyo/bc_learning_sound))


## Our reproduction
From the original study, we have tried to reproduce selected parts, not the entire study. We have trained an __11-layer CNN__ to do __standard, BC__ and __BC+__ learning on the __CIFAR-10__ dataset.

### Data augmentation
We have trained an __11-layer CNN__ to do __standard__ and __BC__ learning without data augmentation on the __CIFAR-10__ dataset.

### Ablation analysis
We have trained an __11-layer CNN__ on the __CIFAR-10__ dataset to do all suggested ablation analysis in the original study (Mixing method, Label, # mixed classes, Where to mix).


## Pytorch
The original study created their model using [Chainer](https://chainer.org/). We ported this code to [Pytorch](https://pytorch.org/). Therefore, the setup for running our model is slightly different.

## Setup
- Install [Pytorch](https://pytorch.org/) on a machine with CUDA GPU, or use [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).
- Run `download_cifar10.py` to download the CIFAR-10 dataset.


## Training
 - Example commands:
	- Standard learning on CIFAR-10:

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/
	

	- BC learning on CIFAR-10:

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/ --BC
	
	- BC+ learning on CIFAR-10:

			python main.py --dataset cifar10 --netType convnet --data path/to/dataset/directory/ --BC --plus
	
- Notes:
	- By default, the training runs 10 times. You can specify the number of trials by using --nTrials command.
	- Please check [opts.py](https://github.com/RickardKarl/bc_learning_image/blob/bc_learning_pytorch/opts.py) for other command line arguments.
	- Please checkout to the respective branch for ablation analysis training.

## Results

Error rate for 11-layer CNN on CIFAR-10

| Learning | Original study | Our result |
|:--|:-:|:-:|
| Standard | 6.07  | 6.59|
| BC | 5.40 | 5.67|
| BC+| **5.22** | **5.51** |

- For more of our results, please see our [blog post](https://bbbaiqian.github.io/CS4240_project)
- For more of the results from the original study, please see the paper: [Between-class Learning for Image Classification](https://arxiv.org/abs/1711.10284)

## Caltech101
We also trained an 11-layer CNN on the Caltech-101 dataset. Please checkout branch `caltech101_experiment` for the code used for training.

---
