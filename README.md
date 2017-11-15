# A PyTorch Implementation of "Riemannian approach to batch normalization"

A PyTorch implementation of **Riemannian approach to batch normalization** (NIPS 2017) by Minhyung Cho and Jaehyung Lee (https://arxiv.org/abs/1709.09603).


Refer to https://github.com/MinhyungCho/riemannian-batch-normalization/ for a tensorflow implementation.

## Train
The commands below are examples.
```
[SGD] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10
[SGD-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2
[Adam-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05
```

***
This repository was forked from a subfolder named "pytorch" in https://github.com/szagoruyko/wide-residual-networks. Below is the readme from the folder.

PyTorch training code for Wide Residual Networks
==========

PyTorch training code for Wide Residual Networks:
http://arxiv.org/abs/1605.07146

The code reproduces *exactly* it's lua version:
https://github.com/szagoruyko/wide-residual-networks


# Requirements

The script depends on opencv python bindings, easily installable via conda:

```
conda install -c menpo opencv3
```

After that and after installing pytorch do:

```
pip install -r requirements.txt
```


# Howto

Train WRN-28-10 on 4 GPUs:

```
python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --ngpu 4 --gpu_id 0,1,2,3
```
