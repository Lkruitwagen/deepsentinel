import glob, yaml, os, json, copy, random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ContrastiveLoss(Dataset):
    """
    A pytorch DataLoader class for generating samples for classifying land cover using S1/S2 imagery

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
    lc_agg: int
        The level of land-cover aggregation in [0,1,2] corresponding to the heirarchical levels:
        https://www.eea.europa.eu/data-and-maps/figures/corine-land-cover-2006-by-country/legend
        
    """

    def __init__(self, data_config, data_dir, bands=None, source='GEE', channel_stats = None, patch_size=64, augmentations = None, random_crop=True, aug_crop=None, s1_dropout=None, s2_dropout=None, warmup_epochs=None, ramp_epochs=None, N_jitters=None, jitter_params=None, start_por=0., end_por=1., seed=None):
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
        self.random_crop=random_crop
        
        self.records = self._parse_records()
        
        # seed the randomiser
        if seed:
            random.seed(seed)
            
        # shuffle the records
        random.shuffle(self.records)
        
        # trim the records for trn/crossval
        start_idx = int(start_por*len(self.records))
        end_idx = int(end_por*len(self.records))
        self.records = self.records[start_idx:end_idx]
        
        self.random_crop=random_crop 
        
        if bands:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands']) if band in bands] \
                        + [{'idx':idx,'band':band,'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands']) if band in bands] 
        else:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands'])] \
                         + [{'idx':idx,'band':band, 'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands'])]
            
        self.output_bands = copy.copy(self.bands)
        
        assert (len(self.bands)>0), 'At least one band must be included.'
        
        self.augmentations=augmentations
        self.warmup_epochs=warmup_epochs
        self.ramp_epochs=ramp_epochs
        self.aug_crop = aug_crop
        self.S1_dropout_bounds=s1_dropout
        self.S2_dropout_bounds=s2_dropout 
        self.N_jitters_bounds=N_jitters 
        self.jitter_params=jitter_params
        
        
        
            
        
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
        df['LC'] = df['f'].str.contains('LCarr')
        
        record_df = pd.merge(
            df[(df['S2']==True)  & (df['source']=='GEE')].rename(columns={'f':'S2_fname'}), 
            df[(df['S1']==True) & (df['source']=='GEE')].rename(columns={'f':'S1_fname'}), 
            how='outer',on='record'
        )
        record_df = pd.merge(record_df, df[(df['LC']==True)].rename(columns={'f':'LC_fname'}), how='left', on='record')[['record','S2_fname','S1_fname', 'LC_fname']]
        
        return record_df.to_dict(orient='records')[:1000]
        
        
    def __len__(self):
        """ Returns the number of records in the dataset. """
        return len(self.records)            
        
    def _epoch_end(self, epoch,max_epochs):
        ## set some params based on epoch
        if 'crop' in self.augmentations and epoch>self.warmup_epochs and epoch<=self.warmup_epochs+self.ramp_epochs:
            self.max_crop_dist = self.aug_crop['min']+int((self.aug_crop['max']-self.aug_crop['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs)
        else:
            self.max_crop_dist = self.aug_crop['min']
            
        if 'dropout' in self.augmentations and epoch>self.warmup_epochs and epoch<=self.warmup_epochs+self.ramp_epochs:
            self.S2_dropout = self.S2_dropout_bounds['min'] + (self.S2_dropout_bounds['max']-self.S2_dropout_bounds['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
            self.S1_dropout = self.S1_dropout_bounds['min'] + (self.S1_dropout_bounds['max']-self.S1_dropout_bounds['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
        else:
            self.S2_dropout = self.S2_dropout_bounds['min']
            self.S1_dropout = self.S1_dropout_bounds['min']
            
        if 'jitter' in self.augmentations and epoch>self.warmup_epochs and epoch<=self.warmup_epochs+self.ramp_epochs:
            self.N_jitters = self.N_jitters_bounds['min']+int((self.N_jitters_bounds['max']-self.N_jitters_bounds['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs)
            self.brightness =self.jitter_params['brightness']['min']+(self.jitter_params['brightness']['max']-self.jitter_params['brightness']['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
            self.contrast =self.jitter_params['contrast']['min']+(self.jitter_params['contrast']['max']-self.jitter_params['contrast']['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
            self.saturation =self.jitter_params['saturation']['min']+(self.jitter_params['saturation']['max']-self.jitter_params['saturation']['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
            self.hue =self.jitter_params['hue']['min']+(self.jitter_params['hue']['max']-self.jitter_params['hue']['min']) * (epoch-self.warmup_epochs)/self.ramp_epochs
        else:
            self.N_jitters = self.N_jitters_bounds['min']
            self.brightness =self.jitter_params['brightness']['min']
            self.contrast =self.jitter_params['contrast']['min']
            self.saturation =self.jitter_params['saturation']['min']
            self.hue =self.jitter_params['hue']['min']
            
        print ('epoch end:', epoch)
        print ('set max_crop_dist', self.max_crop_dist)
        print ('S2_dropout', self.S2_dropout)
        print ('S1_dropout', self.S1_dropout)
        print ('N_jitters', self.N_jitters)
        print ('contrast', self.contrast)
        print ('brightness', self.brightness)
        print ('saturation', self.saturation)
        print ('hue', self.hue)
        
        
    def __getitem__(self, index):
        """
        Gets an individual sample/target pair indexed by 'index'.


        Returns
        -------
        item: np.array, np.array
            The iterator returning pairs of shape: [n_bands, patch_size, patch_size]
        """
        

        
        if self.random_crop:
            OFFX1 = np.random.choice((256-self.patch_size))
            OFFY1 = np.random.choice((256-self.patch_size))
        else:
            OFFX1 = 0
            OFFY1 = 0
            
        if 'crop' in self.augmentations and self.max_crop_dist>0:
            OFFX2 = np.random.choice(range(-1*min(OFFX1,self.max_crop_dist),min(256-self.patch_size-OFFX1,self.max_crop_dist))) + OFFX1
            OFFY2 = np.random.choice(range(-1*min(OFFY1,self.max_crop_dist),min(256-self.patch_size-OFFY1,self.max_crop_dist))) + OFFY1
        else:
            OFFX2 = OFFX1
            OFFY2 = OFFY1
        
        arrs = {}
        
        V1 = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        V2 = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        
        # get the number of S2 bands for offset
        N_S2 = len([band['idx'] for band in self.bands if band['const']=='S2'])
        
        # load data with band dropout
        if 'S2' in set([band['const'] for band in self.bands]):
            arrs['S2'] = np.transpose(np.load(self.records[index]['S2_fname'])['arr'],[2,0,1])
            choose_bands = [(ii_b,band['idx']) for ii_b,band in enumerate(self.bands) if band['const']=='S2' and np.random.rand()>self.S2_dropout]
            if len(choose_bands)>0:
                S2_ii, S2_idx = list(zip(*choose_bands))
                V1[S2_ii,:,:] = arrs['S2'][S2_idx,OFFX1:self.patch_size+OFFX1,OFFY1:self.patch_size+OFFY1]
            choose_bands = [(ii_b,band['idx']) for ii_b,band in enumerate(self.bands) if band['const']=='S2' and np.random.rand()>self.S2_dropout]
            if len(choose_bands)>0:
                S2_ii, S2_idx = list(zip(*choose_bands))
                V2[S2_ii,:,:] = arrs['S2'][S2_idx,OFFX2:self.patch_size+OFFX2,OFFY2:self.patch_size+OFFY2]

        if 'S1' in set([band['const'] for band in self.bands]):
            arrs['S1'] = np.transpose(np.load(self.records[index]['S1_fname'])['arr'], [2,0,1])
            S1_idx = [band['idx'] for band in self.bands if band['const']=='S1']
            choose_bands = [(ii_b,band['idx']) for ii_b,band in enumerate(self.bands) if band['const']=='S1' and np.random.rand()>self.S1_dropout]
            if len(choose_bands)>0:
                S1_ii, S1_idx = list(zip(*choose_bands))
                V1[S1_ii,:,:] = arrs['S1'][S1_idx,OFFX1:self.patch_size+OFFX1,OFFY1:self.patch_size+OFFY1]
            choose_bands = [(ii_b,band['idx']) for ii_b,band in enumerate(self.bands) if band['const']=='S1' and np.random.rand()>self.S1_dropout]
            if len(choose_bands)>0:
                S1_ii, S1_idx = list(zip(*choose_bands))
                V2[S1_ii,:,:] = arrs['S1'][S1_idx,OFFX2:self.patch_size+OFFX2,OFFY2:self.patch_size+OFFY2]
            
        for _ in range(self.N_jitters):
            rand_channels = tuple(np.random.choice(V1.shape[0],3,replace=False))
            V1[rand_channels,:,:] = transforms.ToTensor()(
                                        transforms.ColorJitter(
                                            brightness=self.brightness,
                                            saturation=self.saturation,
                                            contrast=self.contrast,
                                            hue=self.hue
                                        )(transforms.ToPILImage()(torch.from_numpy(V1[rand_channels,:,:])))).numpy()
            rand_channels = tuple(np.random.choice(V2.shape[0],3,replace=False))
            V2[rand_channels,:,:] = transforms.ToTensor()(
                                        transforms.ColorJitter(
                                            brightness=self.brightness,
                                            saturation=self.saturation,
                                            contrast=self.contrast,
                                            hue=self.hue
                                        )(transforms.ToPILImage()(torch.from_numpy(V2[rand_channels,:,:])))).numpy()
            
            
        if self.channel_stats:
            # normalise the data
            for ii_b, band in enumerate(self.bands):
                V1[ii_b,:,:] = (V1[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                V2[ii_b,:,:] = (V2[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                

        return V1, V2