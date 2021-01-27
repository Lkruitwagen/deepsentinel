import numpy as np
import torch

def plot_rgb(input_tensor, N_samples, input_bands, rgb_bands, channel_stats,constellation, reverse_norm=True):
    band_idx = dict(zip([b['band'] for b in input_bands],list(range(len(input_bands)))))
    
    #print (band_idx)
    #print (channel_stats)
    rgb_arr = np.zeros((N_samples,3,input_tensor.shape[-2], input_tensor.shape[-1]))
    for ii_b,b in enumerate(rgb_bands):
        rgb_arr[:,ii_b,:,:] = (input_tensor[:N_samples,band_idx[b],:,:] * channel_stats['std'][b]) + channel_stats['mean'][b]
        
    if constellation=='S2':
        rgb_arr = (rgb_arr/10000).clip(0.,1.)
    elif constellation=='S1':
        #print ('min','max',rgb_arr.min(), rgb_arr.max())
        rgb_arr = ((rgb_arr+50)/60).clip(0.,1.)
        
    return torch.from_numpy(rgb_arr)