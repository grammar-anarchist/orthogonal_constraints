"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    2019 Jun Li
"""

import argparse
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.optim
import torch.utils.data
import cvtransforms as T
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel
import torch.backends.cudnn as cudnn
from resnet import resnet
from vgg import vgg
from sklearn.model_selection import train_test_split
from torchvision import transforms

import grassmann_optimizer
import stiefel_optimizer
from gutils import unit, qr_retraction

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='./', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lrg', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--bnDecay', default=0, type=float)
parser.add_argument('--omega', default=0.1, type=float)
parser.add_argument('--grad_clip', default=0.1, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='Cayley_SGD', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--const', default=0.1, type=float)
parser.add_argument('--low', default=0.05, type=float)
parser.add_argument('--high', default=0.15, type=float)
parser.add_argument('--hh', default=16, type=int)
parser.add_argument('--hh_multiplier', default=-1.0, type=float)
parser.add_argument('--hh_init', default='normal', type=str)
parser.add_argument('--change_method_epoch', default=-1, type=int)
parser.add_argument('--new_optim_method', default='Householder', type=str)
parser.add_argument('--triv', default='expm', type=str)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

def create_dataset_CIFAR(opt, mode):
    if opt.dataset == 'CIFAR10':
        mean = [125.3, 123.0, 113.9]
        std = [63.0, 62.1, 66.7]
    elif opt.dataset =='CIFAR100':
        mean = [129.3, 124.1, 112.4]
        std = [68.2, 65.4, 70.4]

    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        T.Normalize(mean, std),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
    ])

    train_transform = tnt.transform.compose([
        T.RandomHorizontalFlip(),
        T.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        T.RandomCrop(32),
        convert,
    ])
    ds = getattr(datasets, opt.dataset)(opt.dataroot, train=mode, download=True)
    ds = tnt.dataset.TensorDataset([getattr(ds, 'data'),
                                    getattr(ds, 'targets')])
    return ds.transform({0: train_transform if mode else convert})

def create_dataset_Caltech(opt):
    if opt.dataset == 'Caltech256':
        mean = [0.4915449, 0.48238277, 0.44663385]
        std = [0.24713327, 0.24343619, 0.26151007]
    def to_RGB(x):
        if x.size(dim=0) == 1:
            x = torch.squeeze(x)
            return torch.stack([x,x,x],0)
        return x
    convert = tnt.transform.compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        lambda x: to_RGB(x),
        transforms.Normalize(mean, std)
    ])

    train_transform = tnt.transform.compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32)
    ])
    ds = getattr(datasets, opt.dataset)(opt.dataroot, download=True)
    data = []
    targets = []
    for i in range(len(ds)):
        image, target = ds[i]
        image = convert(image)
        data.append(image)
        targets.append(target)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size = 0.17, shuffle=True, random_state=32)
    ds_train = tnt.dataset.TensorDataset([X_train, y_train])
    ds_test = tnt.dataset.TensorDataset([X_test, y_test])
    return ds_train.transform({0: train_transform}), ds_test

def create_dataset(opt):
    if opt.dataset == 'CIFAR10' or opt.dataset == 'CIFAR100':
        ds_train = create_dataset_CIFAR(opt, True)
        ds_test = create_dataset_CIFAR(opt, False)
        num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    elif opt.dataset == 'Caltech256':
        ds_train, ds_test = create_dataset_Caltech(opt)
        num_classes = 257

    train_loader = ds_train.parallel(batch_size=opt.batchSize, shuffle=True,
                           num_workers=opt.nthread, pin_memory=True)
    test_loader = ds_test.parallel(batch_size=opt.batchSize, shuffle=False,
                           num_workers=opt.nthread, pin_memory=True)
    return train_loader, test_loader, num_classes

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers (???)
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    train_loader, test_loader, num_classes = create_dataset(opt)
    
    # model specific
    if opt.model == 'resnet':
        model = resnet
    elif opt.model == 'vgg':
        model = vgg
    f, params, stats = model(opt.depth, opt.width, num_classes)

    key_g = []
    if opt.optim_method in ['SGDG', 'AdamG', 'Cayley_SGD', 'Cayley_Adam', \
                                'Simple_Cayley', 'Random_Cayley', \
                                'Householder', 'Exponential'] :
        param_g = []
        param_e0 = []
        param_e1 = []

        for key, value in params.items():
            if 'conv' in key and value.size()[0] <= np.prod(value.size()[1:]):
                param_g.append(value)
                key_g.append(key)
                if opt.optim_method in ['SGDG', 'AdamG']:
                    unitp, _ = unit(value.data.view(value.size(0), -1)) 
                    value.data.copy_(unitp.view(value.size()))
                elif opt.optim_method in ['Cayley_SGD', 'Cayley_Adam', 'Simple_Cayley', 'Random_Cayley']:
                    q = qr_retraction(value.data.view(value.size(0), -1)) 
                    value.data.copy_(q.view(value.size()))               
            elif 'bn' in key or 'bias' in key:
                param_e0.append(value)
            else:
                param_e1.append(value)

    def create_optimizer(opt, lr, lrg):
        print('creating optimizer with lr = ', lr, ' lrg = ', lrg)
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(params.values(), lr, 0.9, weight_decay=opt.weightDecay)

        elif opt.optim_method == 'SGDG':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'grassmann':True, 'omega':opt.omega, 'grad_clip':opt.grad_clip}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return grassmann_optimizer.SGDG([dict_g, dict_e0, dict_e1])

        elif opt.optim_method == 'AdamG':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'grassmann':True, 'omega':opt.omega, 'grad_clip':opt.grad_clip}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return grassmann_optimizer.AdamG([dict_g, dict_e0, dict_e1])
        
        elif opt.optim_method == 'Cayley_SGD':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.SGDG([dict_g, dict_e0, dict_e1])
        
        elif opt.optim_method == 'Cayley_Adam':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.AdamG([dict_g, dict_e0, dict_e1])

        elif opt.optim_method == 'Simple_Cayley':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True,'const':opt.const}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.SC([dict_g, dict_e0, dict_e1])
        
        elif opt.optim_method == 'Random_Cayley':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True,'low':opt.low,'high':opt.high}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.RC([dict_g, dict_e0, dict_e1])

        elif opt.optim_method == 'Householder':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.Householder([dict_g, dict_e0, dict_e1], U_init=opt.hh_init, hh=opt.hh, hh_multiplier=opt.hh_multiplier)

        elif opt.optim_method == 'Exponential':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.Exponential([dict_g, dict_e0, dict_e1], triv=opt.triv)


    optimizer = create_optimizer(opt, opt.lr, opt.lrg)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in list(params.items()):
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data), end='')
        print(' on G(1,n)' if key in key_g else '')

    print('\nAdditional buffers:')
    kmax = max(len(key) for key in stats.keys())
    for i, (key, v) in enumerate(stats.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v))

    n_training_params = sum(p.numel() for p in params.values())
    n_parameters = sum(p.numel() for p in params.values()) + sum(p.numel() for p in stats.values())
    print('Total number of parameters:', n_parameters, '(%d)'%n_training_params)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        y = data_parallel(f, inputs, params, stats, sample[2], list(range(opt.ngpu)))
        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in list(params.items())},
                        stats=stats,
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'wb'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data.item())

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch == opt.change_method_epoch:
            power=sum(epoch>=i for i in epoch_step)
            lr = opt.lr*pow(opt.lr_decay_ratio, power)
            lrg = opt.lrg*pow(opt.lr_decay_ratio, power)
            opt.optim_method = opt.new_optim_method
            state['optimizer'] = create_optimizer(opt, lr, lrg)
        elif epoch in epoch_step:
            power=sum(epoch>=i for i in epoch_step)
            lr = opt.lr*pow(opt.lr_decay_ratio, power)
            lrg = opt.lrg*pow(opt.lr_decay_ratio, power)
            if opt.optim_method == 'Householder' or opt.optim_method == 'Exponential':
                state['optimizer'] = state['optimizer'].reset(lr, lrg)
            else:
                state['optimizer'] = create_optimizer(opt, lr, lrg)


    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
