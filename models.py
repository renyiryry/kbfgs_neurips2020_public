import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Model_class(nn.Module):
    def __init__(self, params):
        super(Model_class, self).__init__()
        name_dataset = params['name_dataset']
        self.name_loss = params['name_loss']

        if name_dataset in ['MNIST', 'CURVES', 'FACES']:
            self.name_model = 'fully-connected'
        else:
            print('Error: unkown dataset')
            sys.exit()

        if self.name_model == 'fully-connected': 
            if name_dataset == 'MNIST':
                layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
                self.activations_all = ['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear']

            elif name_dataset == 'CURVES':
                layersizes = [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]
                self.activations_all = ['relu', 'relu', 'relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
            
            elif name_dataset == 'FACES':
                layersizes = [625, 2000, 1000, 500, 30, 500, 1000, 2000, 625]
                self.activations_all = ['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear']
            else:
                print('Dateset not supported!')
                sys.exit()
        else:
            print('Error: model name not yet supported.')
            sys.exit()
            
        self.layersizes = layersizes
        layers_params = get_layers_params(self.name_model, layersizes, self.activations_all, params)
        self.layers_all = []

        for l in range(len(layers_params)):
            if layers_params[l]['name'] == 'fully-connected':
                self.layers_all.append(
                nn.Linear(layers_params[l]['input_size'], layers_params[l]['output_size'], bias=True)
                )        
            else:
                print('Error: layer unsupported for ' + layers_params[l]['name'])
                sys.exit()

        self.layers_weight = []
        for l in range(len(layers_params)):
            if layers_params[l]['name'] in ['fully-connected']:
                layers_weight_l = {}
                layers_weight_l['W'] = self.layers_all[l].weight
                layers_weight_l['b'] = self.layers_all[l].bias
                self.layers_weight.append(layers_weight_l)
            else:
                print('Error: layer unsupported when define weight for ' + layers_params[l]['name'])
                sys.exit()
   
        self.layers_params_all = layers_params
        layers_params = []
        self.layers = []
        for l in range(len(self.layers_params_all)):
            if self.layers_params_all[l]['name'] in ['fully-connected']:
                layers_params.append(self.layers_params_all[l])
                self.layers.append(self.layers_all[l])
            else:
                print('error: unkown layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
                
        self.layers_params = layers_params
        self.numlayers = len(layers_params)
        self.numlayers_all = len(self.layers_params_all)
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        a = []
        h = []
        input_ = x

        for l in range(self.numlayers_all):
            if self.layers_params_all[l]['name'] in ['fully-connected']:
                h.append(input_)
                input_, a_l = get_layer_forward(input_, self.layers_all[l], self.layers_params_all[l]['activation'], self.layers_params_all[l])
                a.append(a_l)
            else:
                print('error: unknown self.layers_params_all[l][name]: ' + self.layers_params_all[l]['name'])
                sys.exit()
        return input_, a, h

def get_layers_params(name_model, layersizes, activations, params):
    if name_model == 'fully-connected':
        layers_ = []
        for l in range(len(layersizes) - 1):
            layer_i = {}
            layer_i['name'] = 'fully-connected'
            layer_i['input_size'] = layersizes[l]
            layer_i['output_size'] = layersizes[l+1]
            layer_i['activation'] = activations[l]
            layers_.append(layer_i)
    else:
        print('Error: unknown model name in get_layers_params')
        sys.exit()
    return layers_

def get_layer_forward(input_, layer_, activation_, layer_params):
    if layer_params['name'] == 'fully-connected': 
        a_ = layer_(input_)
        h_ = get_post_activation(a_, activation_)
        a_.retain_grad()
    else:
        print('Error: unkown layer')
        sys.exit()
    output_ = h_
    pre_ = a_
    return output_, pre_


def get_post_activation(pre_, activation):
    if activation == 'relu':
        post_ = F.relu(pre_)
    elif activation == 'sigmoid':
        post_ = torch.sigmoid(pre_)
    elif activation == 'linear':
        post_ = pre_
    else:
        print('Error: unsupported activation for ' + activation)
        sys.exit()
    return post_
    
def get_model(params):
    model = Model_class(params)
    if params['if_gpu']:
        model.to(params['device'])
    return model