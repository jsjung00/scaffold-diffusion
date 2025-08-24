
import torch 
import numpy as np 

def get_class_weights(freq):
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    '''
    epsilon_w = 0.001  # eps to avoid zero division
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w)).float()

    return weights

def get_class_freq(train_dl):
    all_values = torch.cat([tens.flatten() for tens in train_dl])
    distribution = torch.bincount(all_values) # counts from 0 to max value. 
    # missing tokenid 252 for structure block which does not exist in training set 
    distribution = distribution.tolist()
    if len(distribution) < 253:
        distribution.extend([0] * (253 - len(distribution)))
    
    return np.array(distribution)