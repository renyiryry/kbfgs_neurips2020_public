import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy
import copy


def get_loss_from_z(model, z, t, reduction):
    if model.name_loss == 'multi-class classification':
        loss = F.cross_entropy(z, t, reduction = reduction)
    elif model.name_loss == 'binary classification':
        loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float().unsqueeze_(1))
        if reduction == 'none':
            loss = loss.squeeze(1)
        
    elif model.name_loss == 'logistic-regression':
        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            loss = loss / z.size(0) / z.size(1)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            
    elif model.name_loss == 'logistic-regression-sum-loss':
        if reduction == 'none':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())
            loss = torch.sum(loss, dim=1)
        elif reduction == 'mean':
            loss = torch.nn.BCEWithLogitsLoss(reduction = 'sum')(z, t.float())
            loss = loss / z.size(0)
            
        elif reduction == 'sum':
            loss = torch.nn.BCEWithLogitsLoss(reduction = reduction)(z, t.float())

    elif model.name_loss == 'linear-regression-half-MSE':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t) / 2
            loss = loss / z.size(0)
        
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t) / 2
            loss = torch.sum(loss, dim=1)
            
    elif model.name_loss == 'linear-regression':
        
        if reduction == 'mean':
            loss = torch.nn.MSELoss(reduction = 'sum')(z, t)
            loss = loss / z.size(0)
            
        elif reduction == 'none':
            loss = torch.nn.MSELoss(reduction = 'none')(z, t)
            loss = torch.sum(loss, dim=1)
    
    return loss

def get_zero_torch(params):
    layers_params = params['layers_params']
    device = params['device']
    
    delta = []
    for l in range(len(layers_params)):
        delta_l = {}
        delta_l['W'] = torch.zeros(layers_params[l]['output_size'], layers_params[l]['input_size'], device=device)
        delta_l['b'] = torch.zeros(layers_params[l]['output_size'], device=device)
        delta.append(delta_l)
        
    return delta

def get_subtract(model_grad, delta, params):
    diff_p = get_zero(params)
    for l in range(params['numlayers']):
        for key in diff_p[l]:
            diff_p[l][key] = np.subtract(model_grad[l][key], delta[l][key])
    return diff_p

def get_subtract_torch(model_grad, delta):
    diff_p = []
    for l in range(len(model_grad)):
        diff_p_l = {}
        for key in model_grad[l]:
            diff_p_l[key] = torch.sub(model_grad[l][key], delta[l][key])
        diff_p.append(diff_p_l)
    return diff_p



def get_plus(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = np.add(model_grad[l][key], delta[l][key])
        sum_p.append(sum_p_l)
    return sum_p

def get_plus_torch(model_grad, delta):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + delta[l][key]
        sum_p.append(sum_p_l)
    return sum_p

def get_if_nan(p):
    for l in range(len(p)):
        for key in p[l]:
            if torch.sum(p[l][key] != p[l][key]):
                return True
    return False



def get_torch_tensor(p, params):
    p_torch = []
    for l in range(len(p)):
        p_torch_l = {}
        for key in p[l]:
            p_torch_l[key] = torch.from_numpy(p[l][key]).to(params['device'])
        p_torch.append(p_torch_l)
    return p_torch

def get_plus_scalar(alpha, model_grad):
    sum_p = []
    for l in range(len(model_grad)):
        sum_p_l = {}
        for key in model_grad[l]:
            sum_p_l[key] = model_grad[l][key] + alpha
        sum_p.append(sum_p_l)
    return sum_p

def get_multiply_scalar(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_no_grad(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha * delta[l][key].data
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_scalar_blockwise(alpha, delta, params):
    alpha_p = []
    for l in range(params['numlayers']):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = alpha[l] * delta[l][key]
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply_torch(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = torch.mul(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_multiply(alpha, delta):
    alpha_p = []
    for l in range(len(delta)):
        alpha_p_l = {}
        for key in delta[l]:
            alpha_p_l[key] = np.multiply(alpha[l][key], delta[l][key])
        alpha_p.append(alpha_p_l)
    return alpha_p

def get_weighted_sum_batch(hat_v, batch_grads_test, params):
    alpha_p = get_zero(params)
    for l in range(params['numlayers']):
        alpha_p['W'][l] = np.sum(hat_v[:, None, None] * batch_grads_test['W'][l], axis=0)
        alpha_p['b'][l] = np.sum(hat_v[:, None] * batch_grads_test['b'][l], axis=0)
    return alpha_p

def get_opposite(delta):
    numlayers = len(delta)
    p = []
    for l in range(numlayers):
        p_l = {}
        for key in delta[l]:
            p_l[key] = -delta[l][key]
        p.append(p_l)
        
    return p

def get_model_grad(model, params):
    model_grad_torch = []
    for l in range(model.numlayers):
        model_grad_torch_l = {}
        for key in model.layers_weight[l]:
            model_grad_torch_l[key] = copy.deepcopy(model.layers_weight[l][key].grad)
        model_grad_torch.append(model_grad_torch_l)
    return model_grad_torch


def get_regularized_loss_and_acc_from_x_whole_dataset(model, x, t, reduction, params):
    N1 = params['N1']
    N1 = np.minimum(N1, len(x))
    
    i = 0
    device = params['device']
    
    list_loss = []
    list_acc = []
    
    model.eval()
    
    while i + N1 <= len(x):
        with torch.no_grad():
            z, test_a, test_h = model.forward(torch.from_numpy(x[i: i+N1]).to(device))
            
        torch_t_mb = torch.from_numpy(t[i: i+N1]).to(params['device'])
        list_loss.append(
            get_regularized_loss_from_z(model, z, torch_t_mb,
                reduction, params['tau']).item())
        list_acc.append(
            get_acc_from_z(model, params, z, torch_t_mb))
       
        i += N1
    model.train()
    
    return sum(list_loss) / len(list_loss), sum(list_acc) / len(list_acc)

def get_regularized_loss_from_z(model, z, t, reduction, tau):
    loss = get_loss_from_z(model, z, t, reduction)
    loss += 0.5 * tau *\
    get_dot_product_torch(model.layers_weight, model.layers_weight)
    return loss

def get_if_stop(args, i, iter_per_epoch, timesCPU):
    if args['if_max_epoch']:
        if i < int(args['max_epoch/time'] * iter_per_epoch):
            return False
        else:
            return True
    else:
        if timesCPU[-1] < args['max_epoch/time']:
            return False
        else:
            return True
        
        
def get_square(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.square(delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_square_torch(delta_1):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.mul(delta_1[l][key], delta_1[l][key])
        sqaure_p.append(sqaure_p_l)  
    return sqaure_p

def get_sqrt(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_sqrt_torch(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.sqrt(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_max_with_0(delta_1):
    sqaure_p = []
    for l in range(len(delta_1)):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = F.relu(delta_1[l][key])
        sqaure_p.append(sqaure_p_l) 
    return sqaure_p

def get_divide(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = np.divide(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_divide_torch(delta_1, delta_2):
    numlayers = len(delta_1)
    sqaure_p = []
    for l in range(numlayers):
        sqaure_p_l = {}
        for key in delta_1[l]:
            sqaure_p_l[key] = torch.div(delta_1[l][key], delta_2[l][key])
        sqaure_p.append(sqaure_p_l)
    return sqaure_p

def get_dot_product_torch(delta_1, delta_2):
    dot_product = 0
    for l in range(len(delta_1)):
        for key in delta_1[l]:
            dot_product += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
    return dot_product

def get_dot_product_blockwise_torch(delta_1, delta_2):
    dot_product = []
    for l in range(len(delta_1)):
        dot_product_l = 0
        for key in delta_1[l]:
            dot_product_l += torch.sum(torch.mul(delta_1[l][key], delta_2[l][key]))
        dot_product.append(dot_product_l)
    return dot_product

def get_dot_product_batch(model_grad, batch_grads_test, params):
    # numlayers = params['numlayers']
    
    dot_product = np.zeros(len(batch_grads_test['W'][0]))
    for l in range(params['numlayers']):
        dot_product += np.sum(
            np.sum(np.multiply(model_grad['W'][l][None, :], batch_grads_test['W'][l]), axis=-1), axis=-1)
        dot_product += np.sum(np.multiply(model_grad['b'][l][None, :], batch_grads_test['b'][l]), axis=-1)
    
    return dot_product

def get_acc_from_z(model, params, z, torch_t):
    if model.name_loss == 'multi-class classification':
        y = z.argmax(dim=1)
        acc = torch.mean((y == torch_t).float())
        
    elif model.name_loss == 'binary classification':
        z_1 = torch.sigmoid(z)
        y = (z_1 > 0.5)
        y = y[:, 0]
        acc = np.mean(y.cpu().data.numpy() == np_t)
    elif model.name_loss in ['logistic-regression',
                             'logistic-regression-sum-loss']:
        z_sigmoid = torch.sigmoid(z)
        criterion = nn.MSELoss(reduction = 'mean')
        acc = criterion(z_sigmoid, torch_t)
  
    elif model.name_loss in ['linear-regression',
                             'linear-regression-half-MSE']:
        acc = nn.MSELoss(reduction = 'mean')(z, torch_t)

    
    acc = acc.item()
    
    return acc

def get_homo_grad(model_grad_N1, params):
    device = params['device']

    homo_model_grad_N1 = []
    for l in range(params['numlayers']):
        homo_model_grad_N1_l = torch.cat((model_grad_N1[l]['W'], model_grad_N1[l]['b'].unsqueeze(1)), dim=1)
        homo_model_grad_N1.append(homo_model_grad_N1_l)

    return homo_model_grad_N1  
    
