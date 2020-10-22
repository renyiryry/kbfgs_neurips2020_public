import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import scipy
import copy

import time
import pickle


import os
import math
import psutil
import itertools
import datetime
import shutil


from functions_utils import *

def train_initialization(data_, params, args):
    algorithm = params['algorithm']
    
    params['N1'] = args['N1']
    params['N2'] = args['N2']
    
    if algorithm in ['KFAC']:
        params['kfac_damping_lambda'] = args['kfac_damping_lambda']

        device = params['device']
        layersizes = params['layersizes']
        numlayers = params['numlayers']

        A = []  # KFAC A
        G = []  # KFAC G

        for l in range(numlayers):
            A.append(torch.zeros(layersizes[l] + 1, layersizes[l] + 1, device=device))
            G.append(torch.zeros(layersizes[l+1], layersizes[l+1], device=device))
            
        data_['A'] = A
        data_['G'] = G
        A_inv, G_inv = numlayers * [0], numlayers * [0]

        data_['A_inv'] = A_inv
        data_['G_inv'] = G_inv

        params['kfac_inverse_update_freq'] = args['kfac_inverse_update_freq']
        params['kfac_rho'] = args['kfac_rho']
        
        N1 = params['N1']
        model = data_['model']
        if N1 < params['num_train_data']:        
            i = 0 # position of training data
            j = 0 # position of mini-batch

            while i + N1 <= params['num_train_data']:
                X_mb, _ = data_['dataset'].train.next_batch(N1)
                X_mb = torch.from_numpy(X_mb).to(device)
                z, a, h = model.forward(X_mb)
                params['N2_index'] = list(range(N1))
                
                t_mb_pred = sample_from_pred_dist(z, params)
                del params['N2_index']
        
                loss = get_loss_from_z(model, z, t_mb_pred, reduction='mean') # not regularized
                
                model.zero_grad()
                loss.backward()

                i += N1
                j += 1

                for l in range(numlayers):
                    homo_h_l = torch.cat((h[l], torch.ones(N1, 1, device=device)), dim=1)
                    A_j = 1/N1 * torch.mm(homo_h_l.t(), homo_h_l).data
                    data_['A'][l] *= (j-1)/j
                    data_['A'][l] += 1/j * A_j
                    
                    G_j = N1 * torch.mm(a[l].grad.t(), a[l].grad).data
                    data_['G'][l] *= (j-1)/j
                    data_['G'][l] += 1/j * G_j
                    
    elif algorithm in ['RMSprop']:
        params['RMSprop_epsilon'] = args['RMSprop_epsilon']
        data_['RMSprop_momentum_2'] = get_zero_torch(params)
        
        N1 = params['N1']
        device = params['device']
        model = data_['model']
        if N1 < params['num_train_data']:
            i = 0 # position of training data
            j = 0 # position of mini-batch

            while i + N1 <= params['num_train_data']:
                X_mb, t_mb = data_['dataset'].train.next_batch(N1)
                X_mb = torch.from_numpy(X_mb).to(device)
                t_mb = torch.from_numpy(t_mb).to(device)

                z, a, h = model.forward(X_mb)
                loss = get_loss_from_z(model, z, t_mb, reduction='mean') # not regularized

                model.zero_grad()
                loss.backward()

                model_grad = get_model_grad(model, params)
                model_grad = get_plus_torch(
                model_grad,
                get_multiply_scalar_no_grad(params['tau'], model.layers_weight)
                )

                i += N1
                j += 1
                data_['RMSprop_momentum_2'] = get_multiply_scalar(
                        (j-1)/j, data_['RMSprop_momentum_2']
                    )

                data_['RMSprop_momentum_2'] = get_plus_torch(
                        data_['RMSprop_momentum_2'],
                        get_multiply_scalar(1/j, get_square_torch(model_grad))
                    )

    
    elif algorithm in ['K-BFGS', 'K-BFGS(L)']:
        params['Kron_BFGS_A_decay'] = args['Kron_BFGS_A_decay'] 
        params['Kron_LBFGS_Hg_initial'] = args['Kron_LBFGS_Hg_initial'] 
        params['Kron_BFGS_action_h'] = 'Hessian-action-BFGS' 
        params['Kron_BFGS_A_LM_epsilon'] = args['Kron_BFGS_A_LM_epsilon'] 
        params['Kron_BFGS_H_epsilon'] = args['Kron_BFGS_H_epsilon'] 

        params['Kron_BFGS_if_homo'] = True 
        
        if algorithm == 'K-BFGS':
            params['Kron_BFGS_H_initial'] = args['Kron_BFGS_H_initial'] # B           
            params['Kron_BFGS_action_a'] = 'BFGS' # B 
        
        if algorithm == 'K-BFGS(L)':
            params['Kron_BFGS_action_a'] = 'LBFGS' # L
            params['Kron_BFGS_number_s_y'] = args['Kron_BFGS_number_s_y'] # L

        data_['Kron_LBFGS_s_y_pairs'] = {}
        if params['Kron_BFGS_action_a'] == 'LBFGS':
            L = len(params['layersizes']) - 1
            data_['Kron_LBFGS_s_y_pairs']['a'] = []
            for l in range(L):
                data_['Kron_LBFGS_s_y_pairs']['a'].append(
                    {'s': [], 'y': [], 'R_inv': [], 'yTy': [], 'D_diag': [], 'left_matrix': [], 'right_matrix': [], 'gamma': []}
                )
        
               
        layersizes = params['layersizes']
        layers_params = params['layers_params']
        
        device = params['device']
        N1 = params['N1']
        numlayers = params['numlayers']
        model = data_['model']
            
        data_['Kron_BFGS_momentum_s_y'] = []
        for l in range(numlayers):
            Kron_BFGS_momentum_s_y_l = {}
            Kron_BFGS_momentum_s_y_l['s'] = torch.zeros(layersizes[l+1], device=device)
            Kron_BFGS_momentum_s_y_l['y'] = torch.zeros(layersizes[l+1], device=device)
            data_['Kron_BFGS_momentum_s_y'].append(Kron_BFGS_momentum_s_y_l)
 
        data_['Kron_BFGS_matrices'] = []
        for l in range(numlayers):
            Kron_BFGS_matrices_l = {}
            size_A = layers_params[l]['input_size'] + 1
            Kron_BFGS_matrices_l['A'] = torch.zeros(size_A, size_A, device=device, requires_grad=False)
            data_['Kron_BFGS_matrices'].append(Kron_BFGS_matrices_l)

        if params['N1'] < params['num_train_data']:        
            i = 0 
            j = 0 
            while i + N1 <= params['num_train_data']:
                torch.cuda.empty_cache()
                X_mb, t_mb = data_['dataset'].train.next_batch(N1)
                X_mb = torch.from_numpy(X_mb).to(device)
                z, a, h = model.forward(X_mb)

                i += N1
                j += 1

                for l in range(numlayers):
                    homo_h_l = torch.cat((h[l], torch.ones(N1, 1, device=device)), dim=1)
                    A_j = 1/N1 * torch.mm(homo_h_l.t(), homo_h_l).data
                    data_['Kron_BFGS_matrices'][l]['A'] *= (j-1)/j
                    data_['Kron_BFGS_matrices'][l]['A'] += 1/j * A_j
    elif algorithm == 'Adam':
        params['RMSprop_epsilon'] = args['RMSprop_epsilon']
        data_['RMSprop_momentum_2'] = get_zero_torch(params)
        
    return data_, params


def sample_from_pred_dist(z, params):
    name_loss = params['name_loss']
    N2_index = params['N2_index']

    if name_loss == 'multi-class classification':
        from torch.utils.data import WeightedRandomSampler
        pred_dist_N2 = F.softmax(z[N2_index], dim=1)

        t_mb_pred_N2 = list(WeightedRandomSampler(pred_dist_N2, 1))
        t_mb_pred_N2 = torch.tensor(t_mb_pred_N2)
        t_mb_pred_N2 = t_mb_pred_N2.squeeze(dim=1)
  
    elif name_loss == 'binary classification':
        pred_dist_N2 = torch.sigmoid(a[-1][N2_index]).cpu().data.numpy()
        t_mb_pred_N2 = np.random.binomial(n=1, p=pred_dist_N2)
        t_mb_pred_N2 = np.squeeze(t_mb_pred_N2, axis=1)
        t_mb_pred_N2 = torch.from_numpy(t_mb_pred_N2).long()

    elif name_loss in ['logistic-regression',
                       'logistic-regression-sum-loss']:
        pred_dist_N2 = torch.sigmoid(z[N2_index]).data
        t_mb_pred_N2 = torch.distributions.Bernoulli(pred_dist_N2).sample()
        t_mb_pred_N2 = t_mb_pred_N2
    
    elif name_loss == 'linear-regression':
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1/2).sample()
    
    elif name_loss == 'linear-regression-half-MSE':
        t_mb_pred_N2 = torch.distributions.Normal(loc=z[N2_index], scale=1).sample()
    
        
    t_mb_pred_N2 = t_mb_pred_N2.to(params['device'])
    return t_mb_pred_N2

def get_second_order_caches(z, a, h, data_, params):
    if params['if_second_order_algorithm']:
        N1 = params['N1']
        N2 = params['N2']

        N2_index = np.random.permutation(N1)[:N2]
        params['N2_index'] = N2_index

        X_mb = data_['X_mb']

        data_['X_mb_N1'] = X_mb
        X_mb_N2 = X_mb[N2_index]
        data_['X_mb_N2'] = X_mb_N2

        matrix_name = params['matrix_name']
        model = data_['model']
        if matrix_name == 'EF':
            t_mb = data_['t_mb']
            data_['t_mb_pred_N2'] = t_mb[N2_index]
            data_['a_grad_N2'] = [N2 * (a_l.grad)[N2_index] for a_l in a]
            data_['h_N2'] = [h_l[N2_index].data for h_l in h]
            data_['a_N2'] = [a_l[N2_index].data for a_l in a]
        elif matrix_name == 'Fisher':    
            t_mb_pred_N2 = sample_from_pred_dist(z, params)
            data_['t_mb_pred_N2'] = t_mb_pred_N2
            z, a_N2, h_N2 = model.forward(X_mb_N2)
            reduction = 'mean'
            loss = get_loss_from_z(model, z, t_mb_pred_N2, reduction) 
            model.zero_grad()
            loss.backward()
            
            
            data_['a_grad_N2'] = [N2 * (a_l.grad) for a_l in a_N2]
            data_['h_N2'] = h_N2

    return data_

def update_parameter(p_torch, model, params):
    numlayers = params['numlayers']
    alpha = params['alpha']
    device = params['device']

    
    for l in range(numlayers):
        if params['layers_params'][l]['name'] in ['fully-connected']:
            model.layers_weight[l]['W'].data += alpha * p_torch[l]['W'].data
            model.layers_weight[l]['b'].data += alpha * p_torch[l]['b'].data
        
    return model