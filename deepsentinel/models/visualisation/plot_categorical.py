import numpy as np
import torch

def plot_categorical(input_tensor, N_samples, legend_palette):
    
    # Assume in B-C-W-H
    cat_arr = np.zeros((N_samples,input_tensor.shape[-2], input_tensor.shape[-1],3))
    
    # can try in B-W-H-3 later if doesn't work.
    #print (cat_arr.shape)
    #print(input_tensor[:N_samples,:,:,:].shape)
    #cat_arr[torch.argmax(input_tensor[:N_samples,:,:,:], dim=1)==0,:] = np.array([0,22,222])
    #print ('torch squeeze shp',torch.squeeze(torch.argmax(input_tensor[:N_samples,:,:,:], dim=1)).shape)
    #print ('cat arr shape',cat_arr.shape)
    
    for ii_k, (kk,vv) in enumerate(legend_palette.items()):
        cat_arr[torch.squeeze(torch.argmax(input_tensor[:N_samples,:,:,:], dim=1))==ii_k,:] = np.array(vv) # will it into broadcast?
    
    return torch.from_numpy(np.transpose(cat_arr,[0,3,1,2]))