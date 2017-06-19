# YellowFin

YellowFin is an auto-tuning optimizer based on momentum SGD **which requires no manual specification of learning rate and momentum**. It measures the objective landscape on-the-fly and tune momentum as well as learning rate using local quadratic approximation.

The implmentation here can be **a drop-in replacement for any optimizer in PyTorch**. It supports ```step``` and ```zero_grad``` functions like any tensorflow optimizer after ```from yellowfin import YFOptimizer```. 

For more technical details, please refer to our paper [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471).

For more usage details, please refer to the inline documentation of ```tuner_utils/yellowfin.py```. Example usage can be found here for [ResNext on CIFAR10](https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/pytorch-cifar/main.py#L91) and [Tied LSTM on PTB](https://github.com/JianGoForIt/YellowFin_Pytorch/blob/master/word_language_model/main.py#L191).

## Setup instructions for experiments
Please clone the master branch and follow the instructions to run YellowFin on ResNext for CIFAR10, Tied LSTM on Penn Treebank for language modeling. The models are adapted from [ResNext repo](https://github.com/kuangliu/pytorch-cifar) and [PyTorch example tied LSTM repo](https://github.com/pytorch/examples/tree/master/word_language_model) respectively. Thanks to the researchers for developing the models. For more experiments on convolutional and recurrent neural networks, please refer to our [tensorflow implementation](https://github.com/JianGoForIt/YellowFin) of YellowFin.

Note YellowFin is tested with PyTorch v0.1.12 for compatibility. It is tested under Python 2.7.

### run CIFAR10/100 ResNets experiments
The experiments on 110 layer ResNet with CIFAR10 and 164 layer ResNet with CIFAR100 can be launched using
```
cd cifar/scripts
python CIFAR10-release.py (for CIFAR10)
python CIFAR100-release.py (for CIFAR10)
```

### run Penn Treebank LSTM experiments
The experiments on multiple-layer LSTM on Penn Treebank can be launched using
```
cd ptb/scripts
python PTB-release.py
```

### Tensorflow implementation
[YellowFin Tensorflow Repo](https://github.com/JianGoForIt/YellowFin)
