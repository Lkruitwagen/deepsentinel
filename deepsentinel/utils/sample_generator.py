import os, json, yaml, zipfile, geojson, urllib, re, io, logging
from pathlib import Path
import pyproj
from functools import partial
from shapely import geometry, ops
from datetime import datetime as dt 
from datetime import timedelta
import numpy as np
import pandas as pd
from importlib import import_module
import multiprocessing as mp
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from deepsentinel.utils.geoutils import *
from deepsentinel.utils.storageutils import GCPClient, AzureClient
from deepsentinel.utils.downloaders import DL_CLC_downloader, DL_downloader, GEE_downloader, OSM_downloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SampleDownloader:
    
    
    def __init__(self, version, destinations, conf=False, multiprocess=False):

        # load config, credentials
        if not conf:
            self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)
        else:
            self.CONFIG = yaml.load(open(conf,'r'),Loader=yaml.SafeLoader)

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)
        
        self.version = version
        
        self.destinations = destinations
        
        
        # load the point df
        self.pts = pd.read_parquet(os.path.join(self.CONFIG['POINTS_ROOT'], self.version+'.parquet'))
        
        # make directory
        if not os.path.exists(os.path.join(self.CONFIG['DATA_ROOT'],self.version)):
            os.makedirs(os.path.join(self.CONFIG['DATA_ROOT'],self.version))
            
            
        if 'azure' in destinations:
            # Azure cannot into make container in multiprocessing
            from deepsentinel.utils.storageutils import AzureClient
            AzureClient(self.CONFIG['azure_path'], version, make_container=True)
       
    """
    def filter_pts(self, src):
        
        check_pts = pts.copy()
         # only run points not run yet
        if 'gcp' in self.destinations:
            
            check_pts['name_root'] = check_pt.apply(lambda pt:
                                                    os.path.join(
                                                        self.CONFIG['DATA_ROOT'],
                                                        version,
                                                        str(pt.name),
                                                        '_'.join(
                                                            [str(idx),
                                                             src.upper(),
                                                             pt[f'{src.upper()}_S2'].split(':')[2][0:10],
                                                             str(pt['lon']), 
                                                             str(pt['lat'])
                                                            ]
                                                        )
                                                    )
       
            
            gcp_client = GCPClient(self.CONFIG['gcp_credentials_path'],self.CONFIG['gcp_storage_bucket'],self.version)
            done_blobs = [bb.name for bb in sc.client.list_blobs(CONFIG['gcp_storage_bucket'], prefix=self.version)]
            
            pd.DataFrame()
            all_blobnames = [bb.name for bb in gcp_client.list_blobs()]
        
        if 'local' in self.destinations:
        
        if 'azure' in destinations:
            raise NotImplementedError
            
            
        azure_client = AzureClient(CONFIG['azure_path'], version, make_container=False)
        
        [bb for bb in self.client.list_blobs(self.bucket,prefix=source_dir)]
        
    """
        
            
            
    def download_samples_DL(self):
        
        # get any log and filter
        if os.path.exists(os.path.join(os.getcwd(),'logs',f'{self.version}_dl.log')):
            with open(os.path.join(os.getcwd(),'logs',f'{self.version}_dl.log'),'r') as f:
                lines = f.readlines()
            done_idx = list(set([int(el) for el in lines]))
        else:
            done_idx = []
        
        args = []
        step = (len(self.pts.loc[~self.pts.index.isin(done_idx)])//self.CONFIG['N_workers'])+1

        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.loc[~self.pts.index.isin(done_idx)].iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.loc[~self.pts.index.isin(done_idx)].iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w,
                 self.destinations
                )

            )

        #for arg in args:
        #DL_downloader(*args[0])
        #    input('-next-->')
            
        with mp.Pool(self.CONFIG['N_workers']) as P:
            results = P.starmap(DL_downloader, args)

        TLs = {}
        for r in results:
            TLs.update(r[0])

        json.dump(TLs, open(os.path.join(self.CONFIG['DATA_ROOT'], self.version,'tiles.json'),'w'))
        
    def download_samples_GEE(self):
        
        # get tiles
        if self.CONFIG['tiles_source'] =='json':
            TLs = json.load(open(os.path.join(self.CONFIG['DATA_ROOT'],self.version, 'tiles.json'),'r'))
        elif self.CONFIG['tiles_source']=='local':
            # try loading tiles from local
            print ('loading tls from local')
            TLs = {int(path.parent.name):json.load(path.open()) for path in Path(os.path.join(self.CONFIG['DATA_ROOT'], self.version)).rglob('*_tile.json')}

            print (f'got TLS, lenkeys: {len(TLs.keys())}')
        else:
            TLs = {kk:None for kk in self.pts.index.values.tolist()}
            
            
        # get any log and filter
        if os.path.exists(os.path.join(os.getcwd(),'logs',f'{self.version}_gee.log')):
            with open(os.path.join(os.getcwd(),'logs',f'{self.version}_gee.log'),'r') as f:
                lines = f.readlines()
            done_idx = list(set([int(el) for el in lines]))
        else:
            done_idx = []
        
        args = []
        step = (len(self.pts.loc[~self.pts.index.isin(done_idx) & self.pts.index.isin(list(TLs.keys()))])//self.CONFIG['N_workers'])+1
        
        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.loc[~self.pts.index.isin(done_idx) & self.pts.index.isin(list(TLs.keys()))].iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.loc[~self.pts.index.isin(done_idx) & self.pts.index.isin(list(TLs.keys()))].iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w,
                 {str(kk):TLs[kk] for kk in self.pts.loc[~self.pts.index.isin(done_idx) & self.pts.index.isin(list(TLs.keys()))].iloc[ii_w*step:(ii_w+1)*step,:].index.values},
                 self.destinations
                )
            )
        
        with mp.Pool(self.CONFIG['N_workers']) as P:
            results = P.starmap(GEE_downloader, args)
        
        #GEE_downloader(self.version, self.pts, self.pts.index.values, self.CONFIG, 0, TLs, self.destinations)
        
    def download_samples_LC(self):
        
        if os.path.exists(os.path.join(self.CONFIG['DATA_ROOT'],self.version, 'tiles.json')):
            TLs = json.load(open(os.path.join(self.CONFIG['DATA_ROOT'],self.version, 'tiles.json'),'r'))
        else:
            TLs = None
            
        args = []
        step = (len(self.pts)//self.CONFIG['N_workers'])+1
        
        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w,
                 {str(kk):TLs[str(kk)] for kk in self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values},
                 self.destinations
                )
            )
        
        with mp.Pool(self.CONFIG['N_workers']) as P:
            results = P.starmap(DL_CLC_downloader, args)
        
    def download_samples_OSM(self):
        
        if os.path.exists(os.path.join(self.CONFIG['DATA_ROOT'],self.version, 'tiles.json')):
            TLs = json.load(open(os.path.join(self.CONFIG['DATA_ROOT'],self.version, 'tiles.json'),'r'))
        else:
            TLs = None
            
        args = []
        step = (len(self.pts)//self.CONFIG['N_workers'])+1
        
        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w,
                 {str(kk):TLs[str(kk)] for kk in self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values},
                 self.destinations
                )
            )
        
        with mp.Pool(self.CONFIG['N_workers']) as P:
            results = P.starmap(OSM_downloader, args)
        

        
        
if __name__=="__main__":
    downloader=SampleDownloader(version='DEMO_unlabelled', destinations=['gcp','azure'])

    downloader.download_samples_DL()
    downloader.download_samples_GEE()
    #downloader.download_samples_LC()
    #downloader.download_samples_OSM()