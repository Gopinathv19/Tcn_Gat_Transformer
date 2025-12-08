import torch
import os
import pickle
import numpy as np
import random
from tqdm import tqdm

def L2forTest(outputs,targets,obs_length):
    '''
    Evaluation.
    information: [N, 3]
    '''
    seq_length = outputs.shape[0]
    error = torch.norm(outputs-targets,p=2,dim=2)
    error_pred_length = error[obs_length-1:]
    error = torch.sum(error_pred_length)
    error_cnt = error_pred_length.numel()
    if error == 0:
        return 0,0,0,0,0,0
    final_error = torch.sum(error_pred_length[-1])
    final_error_cnt = error_pred_length[-1].numel()
    first_erro = torch.sum(error_pred_length[0])
    first_erro_cnt = error_pred_length[0].numel()
    return error.item(),error_cnt,final_error.item(),final_error_cnt,first_erro.item(),first_erro_cnt

    
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod