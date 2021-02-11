import glob, yaml, os, json, copy

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Tile2VecLoader(Dataset):
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

    def __init__(self, data_config, data_dir, redo_neighbours=False, bands=None, source='GEE', channel_stats = None, patch_size=64):
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
        
        if redo_neighbours:
            ds_name = [el for el in data_dir.split('/') if el!='']
            self.pts = pd.read_parquet(os.path.join(self.data_config['NE_ROOT'],'pts',ds_name[-1]+'.parquet'))
            self._get_neighbours()
        else:
            ds_name = [el for el in data_dir.split('/') if el!='']
            self.pts = pd.read_parquet(os.path.join(self.data_config['NE_ROOT'],'pts',ds_name[-1]+'_neighbours.parquet'))
            self.pts = self.pts.set_index('idx')
            self.pts['neighbours'] = self.pts['neighbours'].apply(json.loads)
            
        
        self.record_df = self._parse_records()
        print ('pts N pre:',len(self.pts))
        
        # points idxs
        self.pts = self.pts[self.pts['neighbours'].str.len()>0]
        print ('pts N 0:',len(self.pts))
        
        # neighbour idxs
        def reduce_neighbours(ll):
            return [el[0] for el in ll]
        
        def verify_neighbours(ll):
            return [el for el in ll if (el[0] in all_intersections)]
        
        for _ in range(3): # iter remove 
            pts_idxs = set(self.pts.index.values.tolist())
        
            self.pts['neighbouridxs'] = self.pts['neighbours'].apply(reduce_neighbours)
            unique_idxs = set(self.pts['neighbouridxs'].explode().unique())

            # record idxs       
            record_idxs = set(self.record_df['record'].values.tolist())

            # get three-set intersections
            all_intersections = unique_idxs & pts_idxs & record_idxs
        
            print('checking neighbours, constraining pts')
            self.pts['neighbours'] = self.pts['neighbours'].apply(verify_neighbours)    # drop neighbours that aren't in 
            self.pts = self.pts[self.pts.index.isin(self.record_df['record'].values.tolist())]   # 
            self.pts = self.pts.loc[(self.pts['neighbours'].str.len()>0)&(self.pts.index.isin(all_intersections)),:]
            self.record_df = self.record_df[self.record_df['record'].isin(all_intersections)]
            #self.pts['neighbours'] = self.pts['neighbours'].apply(verify_neighbours)
            print ('pts N:',_,len(self.pts))
        
        print ('finishing records list')
        # filter records again
        self.record_df = self.record_df[self.record_df['record'].isin(self.pts.index)]
        self.records = self.record_df.to_dict(orient='records')
        self.record_idx = {self.records[ii]['record']:ii for ii in range(len(self.records))}
        self.record_keys = record_idxs
        # drop records 
        
        #print ('records 95628')
        #print (self.records[95628])
        #print ('records_idx 95628')
        #print (self.record_idx[95628])
        #print ('pts 95628')
        
        #print (self.pts.loc[95628,:])
        
        if bands:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands']) if band in bands] \
                        + [{'idx':idx,'band':band,'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands']) if band in bands] 
        else:
            self.bands = [{'idx':idx,'band':band,'const':'S2'} for idx, band in enumerate(self.data_config[source]['S2_bands'])] \
                         + [{'idx':idx,'band':band, 'const':'S1'} for idx, band in enumerate(self.data_config[source]['S1_bands'])]
            
        self.output_bands = copy.copy(self.bands)
        
        assert (len(self.bands)>0), 'At least one band must be included.'
        
    def _get_neighbours(self):
        pts = self.pts.reset_index()
        pts['idx'] = pts['idx'].astype(int)
        
        def get_closeones(row):
            pts['dist_geo'] = np.nan
            pts['dist_euc'] = np.sqrt((pts['lat']-row['lat'])**2 + (pts['lon']-row['lon'])**2)
            pts.loc[pts['dist_euc']<1.,'dist_geo'] = pts.loc[pts['dist_euc']<1.,:].apply(lambda row2: geodesic((row['lat'],row['lon']),(row2['lat'],row2['lon'])).km, axis=1)
            return pts.sort_values('dist_geo').loc[~pts['dist_geo'].isna(),['idx','dist_geo']].values.tolist()[1:]
        
        self.pts['neighbours'] = pts.progress_apply(lambda row: get_closeones(row), axis=1)
           
            
        
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
        
        # filter where there are neighbours
        
        return record_df
        
        
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
        
        arrs = {el:{} for el in ['a','n','d']}
        
        a = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        n = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        d = np.zeros((len(self.bands), self.patch_size, self.patch_size), dtype=np.float32)
        
        ## get neighbour and distant
        
        try:
            n_idxs, n_dists = list(zip(*self.pts.loc[self.records[index]['record'],'neighbours'])) #pts indexed by idx
        except Exception as e:
            print ('index',index)
            print(self.records[index])
            print (e)
            raise
        n_idx = np.random.choice(n_idxs,p=softmax(n_dists))
        d_idx = np.random.choice(list(self.record_keys-set(list(n_idxs)+[self.records[index]['record']])))
        
        S2_idx = []
        S1_idx = []
        
        if 'S2' in set([band['const'] for band in self.bands]):
            arrs['a']['S2'] = np.transpose(np.load(self.records[index]['S2_fname'])['arr'],[2,0,1])
            arrs['n']['S2'] = np.transpose(np.load(self.records[self.record_idx[n_idx]]['S2_fname'])['arr'],[2,0,1])
            arrs['d']['S2'] = np.transpose(np.load(self.records[self.record_idx[d_idx]]['S2_fname'])['arr'],[2,0,1])
            S2_idx = [band['idx'] for band in self.bands if band['const']=='S2']
            a[0:len(S2_idx),:,:] = arrs['a']['S2'][tuple(S2_idx),0:self.patch_size,0:self.patch_size]
            n[0:len(S2_idx),:,:] = arrs['n']['S2'][tuple(S2_idx),0:self.patch_size,0:self.patch_size]
            d[0:len(S2_idx),:,:] = arrs['d']['S2'][tuple(S2_idx),0:self.patch_size,0:self.patch_size]

        if 'S1' in set([band['const'] for band in self.bands]):
            arrs['a']['S1'] = np.transpose(np.load(self.records[index]['S1_fname'])['arr'], [2,0,1])
            arrs['n']['S1'] = np.transpose(np.load(self.records[self.record_idx[n_idx]]['S1_fname'])['arr'], [2,0,1])
            arrs['d']['S1'] = np.transpose(np.load(self.records[self.record_idx[d_idx]]['S1_fname'])['arr'], [2,0,1])
            S1_idx = [band['idx'] for band in self.bands if band['const']=='S1']
            a[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['a']['S1'][tuple(S1_idx),0:self.patch_size,0:self.patch_size]
            n[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['n']['S1'][tuple(S1_idx),0:self.patch_size,0:self.patch_size]
            d[len(S2_idx):len(S2_idx)+len(S1_idx),:,:] = arrs['d']['S1'][tuple(S1_idx),0:self.patch_size,0:self.patch_size]
            
        if self.channel_stats:
            # normalise the data
            for ii_b, band in enumerate(self.bands):
                a[ii_b,:,:] = (a[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                n[ii_b,:,:] = (n[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
                d[ii_b,:,:] = (d[ii_b,:,:] - self.channel_stats['mean'][band['band']]) / self.channel_stats['std'][band['band']]
        


        return a, n, d


    
    