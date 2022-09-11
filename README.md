## Requirements

The script depends on opencv python bindings, easily installable via conda:

```
conda install -c conda-forge opencv 
```

After that and after installing pytorch do:

```
pip install -r requirements.txt
```

## Train

#### Basic command:

```
python main.py --save ./PATH --model resnet --depth 34 --width 1 --gpu_id 0
```

#### Methods and special parameters for them:

```
--optim_method Cayley_SGD
--optim_method Simple_Cayley 
```
   constant $\tau$ in Simple_cayley is given by --lrg

```
--optim_method Random_Cayley --low 0.075 --high 0.125
--optim_method Exponential --triv 'expm'/'cayley'
--optim_method Householder --hh_init 'normal'/'xavier' --hh 16 / --hh_multiplier 0.5
```
   hh gives constant number of Householder matrices, hh_multiplier gives a part depending on the maximum number of Householder matrices
    

#### Other flags:
```
--dataset CIFAR10 
```
   alternatively Caltech256 and CIFAR100 
```
--lr 0.1
--lrg 0.1
```
   lrg is used for stiefel blocks, lr for everything else
```
--epochs 200
--epoch_step '[60, 120, 160]'
```
   epoch_step determines epochs when lr is reduced
