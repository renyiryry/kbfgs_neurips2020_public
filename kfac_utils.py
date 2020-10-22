import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import math

from functions_utils import *


def kfac_update(data_, params):    
    A = data_['A']
    G = data_['G']
    model = data_['model']
    model_grad = data_['model_grad_used_torch']
    model_grad_N1 = data_['model_grad_torch']

    A_inv = data_['A_inv']
    G_inv = data_['G_inv']
    
    N1 = params['N1']
    N2 = params['N2']
    i = params['i']
    inverse_update_freq = params['kfac_inverse_update_freq']
    
   
    lambda_ = params['kfac_damping_lambda']
    lambda_A = math.sqrt(lambda_)
    lambda_G = math.sqrt(lambda_)
    
    alpha = params['alpha']
    numlayers = params['numlayers']
    kfac_rho = params['kfac_rho']

    device = params['device']
    
    a_grad_N2 = data_['a_grad_N2']
    h_N2 = data_['h_N2']
    
    G_ = []
    A_ = []
        
    
#     rho = min(1-1/(i+1), kfac_rho)
    rho = kfac_rho
    
    homo_model_grad_N1 = get_homo_grad(model_grad_N1, params)
    homo_model_grad = get_homo_grad(model_grad, params)
        
    # Step
    delta = []
    for l in range(numlayers):
        G_.append(1/N2 * torch.mm(a_grad_N2[l].t(), a_grad_N2[l]))
        homo_h = torch.cat((h_N2[l], torch.ones(N2, 1, device=device)), dim=1)
        A_.append(1/N2 * torch.mm(homo_h.t(), homo_h))
       
        # Update running estimates of KFAC
        A[l].data = rho*A[l].data + (1-rho)*A_[l].data
        G[l].data = rho*G[l].data + (1-rho)*G_[l].data

        # Amortize the inverse. Only update inverses every now and then
        if (i % inverse_update_freq == 0 or i <= inverse_update_freq) :
            phi_ = 1
          
            A_l_LM = A[l] + (phi_ * lambda_A) * torch.eye(A[l].size()[0], device=device)
            G_l_LM = G[l] + (1 / phi_ * lambda_G) * torch.eye(G[l].size()[0], device=device)

            A_inv[l] = A_l_LM.inverse()
            G_inv[l] = G_l_LM.inverse()
            
        # compute kfac direction
        homo_delta_l = torch.mm(torch.mm(G_inv[l], homo_model_grad[l]), A_inv[l])

        # store the delta
        delta_l = {}
        if params['layers_params'][l]['name'] == 'fully-connected':
            delta_l['W'] = homo_delta_l[:, :-1]
            delta_l['b'] = homo_delta_l[:, -1]
            
        delta.append(delta_l)
    
    p = get_opposite(delta)   
    data_['A'] = A
    data_['G'] = G

    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
    
    data_['p_torch'] = p
    return data_, params