# anonpets89_ezDPS
Basic implementation of ezDPS appeared in PETs'22: "ezDPS: An Efficient and Zero-Knowledge Machine Learning Inference Pipeline"

The machine learning parts are implemented using Python 3.6, with sklearn package. The zero knowledge proofs for ML pipleline is established on libspartan, which is implemented in Rust. 

## Required Libraries
* Python==3.6
* scikit-learn==1.0.2
* numpy==1.19.5
* rustc==1.56.0-nightly (2f662b140 2021-08-29)
* other required libraries can be found in `Cargo.toml`
## Training and Testing of the ML Algorithms
* Cifar-100 Dataset
The standard dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html. We use the python version of Cifar-100.
For 100-classes, run
```
python cifar100/train.py
```
We provide the well-trained parameters at https://drive.google.com/file/d/1QkLIS4UIKkGBKncBt7fm-GTyIMYTHkT6/view?usp=sharing

* UCR-ECG Dataset and LFW Dataset
The detailed instructions are coming soon.

## The Performance of ezDPS
To build the rust codes, run
```
cargo +nightly build --target x86_64-apple-darwin
```

* A toy example on ECG dataset
We first provide a toy example on ECG dataset.

* Other datasets
The detailed instructions are coming soon.
