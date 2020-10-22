import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import math

from functions_utils import *


def RMSprop_update(data_, params):
    model_grad = data_['model_grad_used_torch']
    algorithm = params['algorithm']
    
    if algorithm in ['Adam']:
        beta_1 = params['momentum_gradient_rho']
        i = params['i']
        model_grad = get_multiply_scalar(1 / (1 - beta_1**(i+1)), model_grad)
     

    epsilon = params['RMSprop_epsilon']
    beta_2 = 0.9
        
    
    data_['RMSprop_momentum_2'] =get_plus_torch(get_multiply_scalar(beta_2, data_['RMSprop_momentum_2']), 
            get_multiply_scalar(1-beta_2, get_square_torch(model_grad)))
    
        
    if algorithm in ['Adam']:        
        i = params['i']
        model_grad_second_moment = get_multiply_scalar(1 / (1 - beta_2**(i+1)), data_['RMSprop_momentum_2'])
        
    elif algorithm == 'RMSprop':
        model_grad_second_moment = data_['RMSprop_momentum_2']
     
    p = get_divide_torch(
            model_grad, 
            get_plus_scalar(epsilon, get_sqrt_torch(model_grad_second_moment)))
    p = get_opposite(p)

    data_['p_torch'] = p
    return data_

def SGD_update(data_, params):
    model_grad = data_['model_grad_used_torch']
    p = get_opposite(model_grad)
    data_['p_torch'] = p

    return data_