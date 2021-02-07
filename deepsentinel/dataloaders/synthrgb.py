import glob, yaml, os, json, random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from deepsentinel.dataloaders.vae import VAEDataloader

class SynthRGBDataloader(VAEDataloader):
    """
    A pytorch DataLoader class for generating samples for finetuning using a synthetic rgb problem

    Arguments
    ---------
    data_config: str
        Path to the data generation yaml file
    data_dir: str 
        Path to the directory containing the samples

    Keywords
    --------
    bands: list(str)
        A list of band str names matching names in source
    source: str
        The source of the imagery in the dataset (e.g. GEE or DL)
    channel_stats: str
        The path to any normalising statistics to use
    patch_size: int
        Size of the tile to extract when _get_arr is called (patch_size, patch_size)
        
    """

    def __init__(self, data_config, data_dir, bands=None, source='GEE', channel_stats = None, patch_size=64, start_por=0., end_por=1., seed=None):
        """
        Instantiate the dataloader, populate the data records
        """
        
        super().__init__(data_config=data_config, data_dir=data_dir, bands=bands, source=source, channel_stats=channel_stats, patch_size=patch_size, start_por=start_por, end_por=end_por, seed=seed)      
        
        band_idx = dict(zip(self.data_config[source]['S2_bands'],list(range(len(self.data_config[source]['S2_bands'])))))
        
        if self.source=='GEE':
            self.output_bands = [{'idx':band_idx[band], 'band':band,'const':'S2'} for band in ['B4','B3','B2']]
        else: # DL
            self.output_bands = [{'idx':band_idx[band], 'band':band,'const':'S2'} for band in ['red','green','blue']]
            
        print('output bands',self.output_bands)
        

        
        
    def __getitem__(self, index):
        """
        Gets an individual sample/target pair indexed by 'index'.


        Returns
        -------
        item: np.array, np.array
            The iterator returning pairs of shape: [n_bands, patch_size, patch_size]
        """
        
        # no crop yet
        
        arrs = {}
        
        X = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        Y = np.zeros((3, self.patch_size, self.patch_size), dtype=np.float32)
        
        
        arrs['S2'] = np.transpose(np.load(self.records[index]['S2_fname'])['arr'],[2,0,1])
        arrs['S1'] = np.transpose(np.load(self.records[index]['S1_fname'])['arr'], [2,0,1])
        
        S2_idx = [band['idx'] for band in self.bands if band['const']=='S2']
        S1_idx = [band['idx'] for band in self.bands if band['const']=='S1']
        rgb_idx = [band['idx'] for band in self.output_bands]
            
        X[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['S1'][tuple(S1_idx),0:self.patch_size,0:self.patch_size]
        Y = arrs['S2'][tuple(rgb_idx),0:self.patch_size,0:self.patch_size]
            
        if self.channel_stats:
            # normalise the data
            for ii_b, band in enumerate(self.bands):
                if band['const']=='S1': # don't normalise the S2 empty channels
                    X[ii_b,:,:] = (X[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                    
            for ii_b, band in enumerate(self.output_bands):
                Y[ii_b,:,:] = (Y[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
        

        return X, Y


    
    