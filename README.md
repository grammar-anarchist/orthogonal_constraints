# A PyTorch Implementation of "Riemannian approach to batch normalization"

A PyTorch implementation of **Riemannian approach to batch normalization** (NIPS 2017) by Minhyung Cho and Jaehyung Lee (https://arxiv.org/abs/1709.09603).


Refer to https://github.com/MinhyungCho/riemannian-batch-normalization/ for a tensorflow implementation.


## Abstract
Batch Normalization (BN) has proven to be an effective algorithm for deep neural network training by normalizing the input to each neuron and reducing the internal covariate shift. The space of weight vectors in the BN layer can be naturally interpreted as a Riemannian manifold, which is invariant to linear scaling of weights. Following the intrinsic geometry of this manifold provides a new learning rule that is more efficient and easier to analyze. We also propose intuitive and effective gradient clipping and regularization methods for the proposed algorithm by utilizing the geometry of the manifold. The resulting algorithm consistently outperforms the original BN on various types of network architectures and datasets.

## Results
Classifiation error rate on CIFAR  (median of five runs):

<table>
    <tr>
    	<th>Dataset</th>
        <th colspan="3">CIFAR-10</th>
        <th colspan="3">CIFAR-100</th>
    </tr>
    <tr align="center">
        <th>Model</th>
        <th>SGD</th>
        <th>SGD-G</th>
        <th>Adam-G</th>        
        <th>SGD</th>
        <th>SGD-G</th>
        <th>Adam-G</th>                
    </tr>
    <tr align="center">
        <th>VGG-13</th>
        <td>5.88</td>
        <td>5.87</td>
        <td>6.05</td>        
        <td>26.17</td>
        <td>25.29</td>
        <td>24.89</td>
    </tr>
    <tr align="center">
        <th>VGG-19</th>
        <td>6.49</td>
        <td>5.92</td>
        <td>6.02</td>    
        <td>27.62</td>
        <td>25.79</td>
        <td>25.59</td>
    </tr>
    <tr align="center">
        <th>WRN-28-10</th>
        <td>3.89</td>
        <td>3.85</td>
        <td>3.78</td>        
        <td>18.66</td>
        <td>18.19</td>
        <td>18.30</td>
    </tr>
    <tr align="center">
        <th>WRN-40-10</th>
        <td>3.72</td>
        <td>3.72</td>
        <td>3.80</td>        
        <td>18.39</td>
        <td>18.04</td>
        <td>17.85</td>
    </tr>
</table>

Classification error rate on SVHN (median of five runs):
<table>
    <tr align="center">
        <th>Model</th>
        <th>SGD</th>
        <th>SGD-G</th>
        <th>Adam-G</th>        
    </tr>
    <tr align="center">
        <th>VGG-13</th>
        <td>1.78</td>
        <td>1.74</td>
        <td>1.72</td>        
    </tr>
    <tr align="center">
        <th>VGG-19</th>
        <td>1.94</td>
        <td>1.81</td>
        <td>1.77</td>        
    </tr>
    <tr align="center">
        <th>WRN-16-4</th>
        <td>1.64</td>
        <td>1.67</td>
        <td>1.61</td>        
    </tr>
    <tr align="center">
        <th>WRN-22-8</th>
        <td>1.64</td>
        <td>1.63</td>
        <td>1.55</td>        
    </tr>
</table>

&nbsp;


| WRN-28-10 on CIFAR10 | WRN-28-10 on CIFAR100 | WRN-22-8 on SVHN |
|:---:|:---:|:---:|
| ![CIFAR10](https://user-images.githubusercontent.com/32380857/31753952-34bb561a-b4cf-11e7-9f3e-780df3765891.png) | ![CIFAR100](https://user-images.githubusercontent.com/32380857/31753955-37b807b4-b4cf-11e7-86f5-5f666ff5091a.png) | ![SVHN](https://user-images.githubusercontent.com/32380857/31753960-3a859d58-b4cf-11e7-8918-031349685cb4.png) |

See https://arxiv.org/abs/1709.09603 for details.

    
## Train
The commands below are examples.
```
[SGD] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10
[SGD-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2
[Adam-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05
```
## To apply this algorithm to your model
[grassmann_optimizer.py](https://github.com/MinhyungCho/riemannian-batch-normalization-pytorch/blob/master/grassmann_optimizer.py) is the main implementation which provides the proposed SGD-G and Adam-G optimizer. [main.py](https://github.com/MinhyungCho/riemannian-batch-normalization-pytorch/blob/master/main.py) includes all the steps to apply the provided optimizers to your model.

1. Collect all the weight parameters which need to be optimized on Grassmann manifold (and initialize them to a unit scale):

    ```python
    key_g = []
    if opt.optim_method == 'SGDG' or opt.optim_method == 'AdamG':
        param_g = []
        param_e0 = []
        param_e1 = []

        for key, value in params.items():
            if 'conv' in key and value.size()[0] < np.prod(value.size()[1:]):
                param_g.append(value)
                key_g.append(key)
                # initlize to scale 1
                unitp, _ = unit(value.data.view(value.size(0), -1)) 
                value.data.copy_(unitp.view(value.size()))
            elif 'bn' in key or 'bias' in key:
                param_e0.append(value)
            else:
                param_e1.append(value)
    ```


2. Create the optimizer with proper parameters:
    ```python
    import grassmann_optimizer
    dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'grassmann':True, 'omega':opt.omega, 'grad_clip':opt.grad_clip}
    dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.bnDecay,'nesterov':True}
    dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.weightDecay,'nesterov':True}
    return grassmann_optimizer.SGDG([dict_g, dict_e0, dict_e1])  # or use AdamG
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
