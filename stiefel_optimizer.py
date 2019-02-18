#from .optimizer import Optimizer, required
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

from gutils import unit
from gutils import gproj
from gutils import clip_by_norm
from gutils import xTy
from gutils import gexp
from gutils import gpt
from gutils import gpt2
from gutils import Cayley_loop
from gutils import qr_retraction
from gutils import stiefel_proj_tan
from utils import matrix_norm_one
import random

import pdb

episilon = 1e-8

class SGDG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'stiefel'. 

        If stiefel is True, the variables will be updated by SGD-G proposed 
        as decorrelated weight matrix.

        If stiefel is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        stiefel (bool, optional): whether to use SGD-G (default: False)

        -- parameters in case stiefel is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case stiefel is True
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']
                           
            for p in group['params']:
                if p.grad is None:
                    continue

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    
                    rand_num = random.randint(1,21)
                    if rand_num==1:
                        unity = qr_retraction(unity)
                    
                    g = p.grad.data.view(p.size()[0],-1)
                       
                    
                    lr = group['lr']
                    
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros(g.t().size())
                        if p.is_cuda:
                            param_state['momentum_buffer'] = param_state['momentum_buffer'].cuda()
                            
                    V = param_state['momentum_buffer']
                    P = torch.eye(V.size()[0]).cuda()-0.5*torch.matmul(unity.t(), unity)
                    V = momentum * V - g.t()   
                    Pf = torch.matmul(P, V)
                    PfX = torch.matmul(Pf, unity)
                    W = PfX - PfX.t()
                    t = 0.9 * 2 / (matrix_norm_one(W) + episilon)                    
                    alpha = min(t, lr)
                    
                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.matmul(W, unity.t()) # n-by-p
                    p.data.copy_(p_new.view(p.size()))
                    V.copy_(V_new)               

                else:
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss

class AdamG(Optimizer):
    r"""This optimizer updates variables with two different routines
        based on the boolean variable 'grassmann'. 

        If grassmann is True, the variables will be updated by Adam-G proposed 
        in 'Riemannian approach to batch normalization'.

        If grassmann is False, the variables will be updated by SGD.
        This routine was taken from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

        -- common parameters
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        grassmann (bool, optional): whether to use Adam-G (default: False)

        -- parameters in case grassmann is False 
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        -- parameters in case grassmann is True
        beta2 (float, optional): the exponential decay rate for the second moment estimates (defulat: 0.99)
        epsilon (float, optional): a small constant for numerical stability (default: 1e-8)
        omega (float, optional): orthogonality regularization factor (default: 0)
        grad_clip (float, optional): threshold for gradient norm clipping (default: None)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, 
                 grassmann=False, beta2=0.99, epsilon=1e-8, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        grassmann=grassmann, beta2=beta2, epsilon=epsilon, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']
            
            for p in group['params']:
                if p.grad is None:
                    continue
            
                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1,21)
                    if rand_num==1:
                        unity = qr_retraction(unity)
                        
                    g = p.grad.data.view(p.size()[0],-1)
                    
                    grad_f = stiefel_proj_tan(unity, g)

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size=p.size()
                        param_state['m_buffer'] = torch.zeros([size[0], int(np.prod(size[1:]))])
                        param_state['v_buffer'] = torch.zeros([1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1*m  + (1.0-beta1)*g # p by n
                    vnew = beta2*v  + (1.0-beta2)*(grad_f * grad_f).sum() 
                    
                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)
                    
                    
                    M = mnew_hat.t()
                    U1 = torch.matmul(M, unity)
                    U = U1 - U1.t()
                    V1 = torch.matmul(unity, M)
                    V2 = V1 - V1.t()
                    V = torch.matmul(unity.t(), torch.matmul(V2, unity))
                    W = (U - 0.5 * V)/vnew_hat.add(epsilon).sqrt()
                    
                    t = 0.9 * 2 / (matrix_norm_one(W) + episilon)                    
                    alpha = min(t, group['lr'])
                    
                    p_new = Cayley_loop(unity.t(), W, mnew.t(), -alpha)

                    p.data.copy_(p_new.view(p.size()))
                    m.copy_(stiefel_proj_tan(unity, mnew))
                    v.copy_(vnew)

                    param_state['beta1_power']*=beta1
                    param_state['beta2_power']*=beta2
                    
                else:
                    momentum = group['momentum']
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss       
