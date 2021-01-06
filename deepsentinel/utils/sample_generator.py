import os, json, yaml, zipfile, geojson, urllib, re, io, logging
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
    
    
    def __init__(self, version, destinations, use_dl, use_gee, multiprocess=False):

        # load config, credentials
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)
        
        self.version = version
        
        self.destinations = destinations
        
        self.use_dl = use_dl
        
        self.use_gee = use_gee
        
        # load the point df
        self.pts = pd.read_parquet(os.path.join(self.CONFIG['DATA_ROOT'], 'pts', self.version+'.parquet'))
        
        # make directory
        if not os.path.exists(os.path.join(self.CONFIG['DATA_ROOT'],self.version)):
            os.makedirs(os.path.join(self.CONFIG['DATA_ROOT'],self.version))
            
            
        if 'azure' in destinations:
            # Azure cannot into make container in multiprocessing
            from deepsentinel.utils.storageutils import AzureClient
            AzureClient(self.CONFIG['azure_path'], version, make_container=True)
        

            
            
    def download_samples_DL(self):
        
        args = []
        step = (len(self.pts)//self.CONFIG['N_workers'])+1

        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w,
                 self.destinations
                )

            )

        with mp.Pool(self.CONFIG['N_workers']) as P:
            results = P.starmap(DL_downloader, args)

        TLs = {}
        for r in results:
            TLs.update(r[0])

        json.dump(TLs, open(os.path.join(self.CONFIG['DATA_ROOT'], self.version,'tiles.json'),'w'))
        
    def download_samples_GEE(self):
        
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
    downloader=SampleDownloader(version='DEMO_unlabelled', destinations=['gcp','azure'], use_dl=True, use_gee=False)

    downloader.download_samples_DL()
    downloader.download_samples_GEE()
    #downloader.download_samples_LC()
    #downloader.download_samples_OSM()