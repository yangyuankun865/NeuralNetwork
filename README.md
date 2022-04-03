# Two Layer Neural Network

Numpy implementation of Neural Network with two layers on MNIST

## Description

This project is an implementation of two layer Neural Network with numpy applied on MNIST dataset, with visualization of loss function, accuracy curve and weights in each layer.

## Getting Started

### Dependencies

* Python 3.8

### Installing

* Download this program through git clone and put it in your repository
* Create file plot for all visualization

### Executing program
* You can refer to this environment setting
```
cd your_direction
conda create -n twolayernn --clone base
conda activate twolayernn
conda install numpy
pip install matplotlib
```

* Run the training code directly to train the model with given hyper-parameters
```
python train.py
```

* Run the searching code to search for hyper-parameters, and save the best parameter in weight.npz
```
python search.py
```

* Run the testing code to utilize weight.npz for testing and weight visualization
```
python test.py
```
