# A PyTorch Implementation of "Riemannian approach to batch normalization"

A PyTorch implementation of **Riemannian approach to batch normalization** (NIPS 2017) by Minhyung Cho and Jaehyung Lee (https://arxiv.org/abs/1709.09603).


Refer to https://github.com/MinhyungCho/riemannian-batch-normalization/ for a tensorflow implementation.


## Abstract
Batch Normalization (BN) has proven to be an effective algorithm for deep neural network training by normalizing the input to each neuron and reducing the internal covariate shift. The space of weight vectors in the BN layer can be naturally interpreted as a Riemannian manifold, which is invariant to linear scaling of weights. Following the intrinsic geometry of this manifold provides a new learning rule that is more efficient and easier to analyze. We also propose intuitive and effective gradient clipping and regularization methods for the proposed algorithm by utilizing the geometry of the manifold. The resulting algorithm consistently outperforms the original BN on various types of network architectures and datasets.

## Results
See https://arxiv.org/abs/1709.09603 for details. Only the results for CIFAR-10 and 100 have been verified.

# Requirements

The script depends on opencv python bindings, easily installable via conda:

```
conda install -c menpo opencv3
```

After that and after installing pytorch do:

```
pip install -r requirements.txt
```

## Train
The commands below are examples.
CIFAR-10:
```
[SGD] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10
[SGD-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2
[Adam-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05
```
CIFAR-100:
```
[SGD] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --dataset CIFAR100
[SGD-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2 --dataset CIFAR100
[Adam-G] python main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05 --dataset CIFAR100
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
