import glob, yaml, os, json, copy, random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class CorineLandCover(Dataset):
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

    def __init__(self, data_config, data_dir, bands=None, source='GEE', channel_stats = None, random_crop=True, patch_size=64, lc_agg=0, start_por=0., end_por=1., seed=None):
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
        
        
        if bands:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands']) if band in bands] \
                        + [{'idx':idx,'band':band,'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands']) if band in bands] 
        else:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands'])] \
                         + [{'idx':idx,'band':band, 'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands'])]
            
        self.output_bands = copy.copy(self.bands)
        
        assert (len(self.bands)>0), 'At least one band must be included.'
        
        self.lc_agg = lc_agg
        
        assert (lc_agg in [0,1,2]), 'lc_add must be one of [0,1,2]'
        
        # set up aggregation levels
        legend = json.load(open(self.data_config['DL_LC']['legend_json'],'r'))
        legend_df = pd.DataFrame(legend).T.reset_index().rename(columns={'index':'idx'})
        
        legend_df['lc_level'] = legend_df['description'].str.split(':').str[0].str[0:self.lc_agg+1]
        self.legend_groups = legend_df.groupby('lc_level')['idx'].apply(lambda g: [int(el) for el in g]).to_dict()
        
        legend_df['r'] = legend_df['color'].str[0]
        legend_df['g'] = legend_df['color'].str[1]
        legend_df['b'] = legend_df['color'].str[2]
        
        legend_palette = legend_df[['lc_level','r','g','b']].groupby('lc_level').mean().astype(int)
        legend_palette['rgb'] = legend_palette[['r','g','b']].values.tolist()
        self.legend_palette = legend_palette['rgb'].to_dict()
        
        
            
        
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
        
        if self.random_crop:
            OFFSET = np.random.choice((256-self.patch_size)//2)
        else:
            OSSET = 0
        
        arrs = {}
        
        X = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        
        # do X
        S2_idx = []
        S1_idx = []
        
        if 'S2' in set([band['const'] for band in self.bands]):
            arrs['S2'] = np.transpose(np.load(self.records[index]['S2_fname'])['arr'],[2,0,1])
            S2_idx = [band['idx'] for band in self.bands if band['const']=='S2']
            X[0:len(S2_idx),:,:] = arrs['S2'][tuple(S2_idx),OFFSET:self.patch_size+OFFSET,OFFSET:self.patch_size+OFFSET]

        if 'S1' in set([band['const'] for band in self.bands]):
            arrs['S1'] = np.transpose(np.load(self.records[index]['S1_fname'])['arr'], [2,0,1])
            S1_idx = [band['idx'] for band in self.bands if band['const']=='S1']
            X[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['S1'][tuple(S1_idx),OFFSET:self.patch_size+OFFSET,OFFSET:self.patch_size+OFFSET]
            
            
        if self.channel_stats:
            # normalise the data
            for ii_b, band in enumerate(self.bands):
                X[ii_b,:,:] = (X[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                
        # do Y
        arrs['LC'] = np.load(self.records[index]['LC_fname'])['arr'].astype(int)
        
        if self.lc_agg==2:
            # do a quick categorical transform
            Y = np.eye(len(self.legend_groups))[arrs['LC'][OFFSET:self.patch_size+OFFSET, OFFSET:self.patch_size+OFFSET]]
            #BCHW -> channels fist
            Y = np.transpose(T,[2,0,1])
        else:
            Y = np.zeros((len(self.legend_groups), self.patch_size, self.patch_size), dtype=np.long) # np.long for categorical
            for ii,(kk,vv) in enumerate(self.legend_groups.items()):
                Y[ii,np.isin(arrs['LC'][OFFSET:self.patch_size+OFFSET, OFFSET:self.patch_size+OFFSET],vv)] = 1

        return X, Y