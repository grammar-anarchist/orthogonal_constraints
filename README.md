# Optimization-on-Stiefel-Manifold-via-Cayley-Transform



## Abstract
This paper is about Riemannian optimization on Stiefel manifold with an important application in enforcing orthonormality on parameters of a deep neural network. We specify an efficient way for estimating the retraction mapping --- i.e., mapping of the tangent vector back to the manifold --- the key challenge in Riemannian optimization due to its computational cost. Specifically, we estimate a smooth curve on Stiefel manifold that connects the previous and next update point in optimization using a novel iterative version of the Cayley transform. With this, we extended  conventional stochastic gradient descent (SGD) and ADAM methods to our two new algorithms Cayley SGD with momentum and Cayley ADAM. Convergence of Cayley SGD is theoretically analyzed, while convergence rates of both algorithms are evaluated in the context of training two standard deep  networks --- VGG and wide Resnet --- for image classification. Our results demonstrate that Cayley SGD  and  Cayley ADAM achieve faster convergence without decreasing classification accuracy of the networks, relative to the baseline SGD and ADAM, as well as existing approaches to enforcing orthogonality of network parameters.


## Requirements

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
[SGD] python main.py --save ./logs/cifar10/SGD/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --gpu_id 0
[SGD-G] python main.py --save ./logs/cifar10/SGDG/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2 --gpu_id 0
[Adam-G] python main.py --save ./logs/cifar10/AdamG/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05 --gpu_id 0
[Cayley-SGD] python main.py --save ./logs/cifar10/CayleySGD/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method Cayley_SGD --lr 0.01 --lrg 0.1 --lr_decay_ratio 0.2 --gpu_id 0
[Cayley-Adam] python main.py --save ./logs/cifar10/CayleyAdam/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method Cayley_Adam --lr 0.01 --lrg 0.05 --lr_decay_ratio 0.2 --gpu_id 0
```
CIFAR-100:
```
[SGD] python main.py --save ./logs/cifar100/SGD/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --dataset CIFAR100 --gpu_id 0
[SGD-G] python main.py --save ./logs/cifar100/SGDG/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method SGDG --lr 0.01 --lrg 0.2 --dataset CIFAR100 --gpu_id 0
[Adam-G] python main.py --save ./logs/cifar100/AdamG/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method AdamG --lr 0.01 --lrg 0.05 --dataset CIFAR100 --gpu_id 0
[Cayley-SGD] python main.py --save ./logs/cifar100/CayleySGD/resnet/depth28width10/Stiefel$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method Cayley_SGD --lr 0.01 --lrg 0.1 --lr_decay_ratio 0.2 --dataset CIFAR100 --gpu_id 0
[Cayley-Adam] python main.py --save ./logs/cifar100/CayleyAdam/resnet/depth28width10/train$RANDOM$RANDOM --model resnet --depth 28 --width 10 --optim_method Cayley_Adam --lr 0.01 --lrg 0.05 --lr_decay_ratio 0.2 --dataset CIFAR100 --gpu_id 0
```

## To apply this algorithm to your model
[stiefel_optimizer.py](https://github.com/JunLi-Galios/Efficient-Riemannian-Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/stiefel_optimizer.py) is the main implementation which provides the proposed Cayley_SGD and Cayley_Adam optimizer. [main.py](https://github.com/JunLi-Galios/Efficient-Riemannian-Optimization-on-Stiefel-Manifold-via-Cayley-Transform/blob/master/main.py) includes all the steps to apply the provided optimizers to your model.

1. Collect all the weight parameters which need to be optimized on Stiefel manifold:

    ```python
    key_g = []
    if opt.optim_method in ['SGDG', 'AdamG', 'Carley_SGD', 'Carley_Adam'] :
        param_g = []
        param_e0 = []
        param_e1 = []

        for key, value in params.items():
            if 'conv' in key and value.size()[0] <= np.prod(value.size()[1:]):
                param_g.append(value)
                key_g.append(key)
                if opt.optim_method in ['SGDG', 'AdamG']:
                    # initlize to scale 1
                    unitp, _ = unit(value.data.view(value.size(0), -1)) 
                    value.data.copy_(unitp.view(value.size()))
                elif opt.optim_method == ['Carley_SGD', 'Carley_Adam']:
                    # initlize to orthogonal matrix
                    q = qr_retraction(value.data.view(value.size(0), -1)) 
                    value.data.copy_(q.view(value.size()))               
            elif 'bn' in key or 'bias' in key:
                param_e0.append(value)
            else:
                param_e1.append(value)
    ```


2. Create the optimizer with proper parameters:
    ```python
    import stiefel_optimizer
    dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
    dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
    dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
    return stiefel_optimizer.SGDG([dict_g, dict_e0, dict_e1])  # or use AdamG
    ```
