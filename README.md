# anonpets89_ezDPS
Basic implementation of ezDPS appeared in PETs'22: "ezDPS: An Efficient and Zero-Knowledge Machine Learning Inference Pipeline"

The machine learning parts are implemented using Python 3.6, with sklearn package. The zero knowledge proofs for ML pipleline is established on libspartan, which is implemented in Rust. 

## Required Libraries

## Training and Testing of the ML Algorithms
* Cifar-100 Dataset
The standard dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html. We use the python version of Cifar-100.
For 100-classes, run
```
python cifar100/train.py
```
We provide the well-trained parameters at https://drive.google.com/file/d/1QkLIS4UIKkGBKncBt7fm-GTyIMYTHkT6/view?usp=sharing

## The Performance of ezDPS
* A toy example on ECG dataset
