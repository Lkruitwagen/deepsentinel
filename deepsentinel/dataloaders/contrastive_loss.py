import glob, yaml, os, json

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class ContrastiveLossDataloader(Dataset):
    """
    A pytorch DataLoader class for generating samples for pretraining a variational autoencoder

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

    def __init__(self, data_config, data_dir, bands=None, source='GEE', channel_stats = None, patch_size=64):
        """
        Instantiate the dataloader, populate the data records
        """
        
        self.data_config = yaml.load(open(data_config, 'r'), Loader = yaml.SafeLoader)
        self.data_dir = data_dir
        self.source=source
        
        if channel_stats:
            self.channel_stats = json.load(open(channel_stats,'r'))
        else:
            self.channel_stats = None
        
        self.patch_size=patch_size
        
        self.records = self._parse_records()
        
        if bands:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands']) if band in bands] \
                        + [{'idx':idx,'band':band,'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands']) if band in bands] 
        else:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands'])] \
                         + [{'idx':idx,'band':band, 'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands'])]

        
        assert (len(self.bands)>0), 'At least one band must be included.'
            
        
    def _parse_records(self):
        
        ### maybe save them as json for quick load
        #if os.path.exists(os.path.join(self.data_dir,'records.json')):
        #    return json.load(open(os.path.join(self.data_dir, 'records.json'),'r'))
        #else:
        #    top_dirs = glob.glob(os.path.join(data_dir,'*/'))
        
        all_files = glob.glob(self.data_dir + '/*/*', recursive=True)
        
        recs = [{
            'record':int(os.path.split(f)[-1].split('_')[0]),
            'source':os.path.split(f)[-1].split('_')[1],
            'f':f
        } for f in all_files]
        
        df = pd.DataFrame(recs)
        
        df['S2'] = df['f'].str.contains('S2arr')
        df['S1'] = df['f'].str.contains('S1arr')
        
        record_df = pd.merge(
            df[(df['S2']==True)  & (df['source']=='GEE')].rename(columns={'f':'S2_fname'}), 
            df[(df['S1']==True) & (df['source']=='GEE')].rename(columns={'f':'S1_fname'}), 
            how='outer',on='record'
        )[['record','S2_fname','S1_fname']]
        
        return record_df.to_dict(orient='records')
        
        
    def __len__(self):
        """ Returns the number of records in the dataset. """
        return len(self.records)            
        
        
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
        
        S2_idx = []
        S1_idx = []
        
        if 'S2' in set([band['const'] for band in self.bands]):
            arrs['S2'] = np.transpose(np.load(self.records[index]['S2_fname'])['arr'],[2,0,1])
            S2_idx = [band['idx'] for band in self.bands if band['const']=='S2']
            X[0:len(S2_idx),:,:] = arrs['S2'][tuple(S2_idx),0:self.patch_size,0:self.patch_size]

        if 'S1' in set([band['const'] for band in self.bands]):
            arrs['S1'] = np.transpose(np.load(self.records[index]['S1_fname'])['arr'], [2,0,1])
            S1_idx = [band['idx'] for band in self.bands if band['const']=='S1']
            X[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['S1'][tuple(S1_idx),0:self.patch_size,0:self.patch_size]
            
            
        if self.channel_stats:
            # normalise the data
            for ii_b, band in enumerate(self.bands):
                X[ii_b,:,:] = (X[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
        
        Y = X.copy()

        return X, Y


    
    