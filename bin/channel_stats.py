""" A command-line script to generate channel-wise stats for datasets."""

import argparse, os, json
import numpy as np

from deepsentinel.dataloaders.vae import VAEDataloader

def generate_channelstats(dataset, source, N):
    
    if not os.path.exists(os.path.join(os.getcwd(),'data','channel_stats')):
        os.makedirs(os.path.join(os.getcwd(),'data','channel_stats'))
        
    loader = VAEDataloader(os.path.join(os.getcwd(),'DATA_CONFIG.yaml'), 
                           dataset, 
                           bands=None, 
                           source=source, 
                           channel_stats = None, 
                           patch_size=64)
    
    _X, _ = loader.__getitem__(0)
    
    assert (N<len(loader)), f'Not enough samples in dataset for {N}'
    
    samples = np.zeros((*_X.shape,N))
    
    for ii,idx in enumerate(np.random.choice(len(loader),N, replace=False)):
        _X, _ = loader.__getitem__(idx)
        samples[...,ii] = _X
        
    channel_stats = {
        'mean':dict(zip([band['band'] for band in loader.bands],np.mean(samples, axis=(1,2,3)).tolist())),
        'std':dict(zip([band['band'] for band in loader.bands],np.std(samples, axis=(1,2,3)).tolist()))
    }

    json.dump(channel_stats, open(os.path.join(os.getcwd(),'data','channel_stats',f'{dataset.split("/")[-2]}_{source}.json'),'w'))


parser = argparse.ArgumentParser(description='A script to sync a local directory with cloud-stored data.')
parser.add_argument('data_dir', type=str, help='The data directy.')
parser.add_argument('source', type=str, help='The dataset source subset.')
parser.add_argument('N', type=int, help='The number of samples from which to derive channel stats.')
if __name__=="__main__":
    args = parser.parse_args()
    
    generate_channelstats(args.data_dir, args.source, args.N)