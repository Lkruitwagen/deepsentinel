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
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt


from deepsentinel.utils.geoutils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def DL_downloader(version, pts, ii_ps, CONFIG, mp_idx):
    
    import descarteslabs as dl
    raster_client = dl.Raster()
    
    logger_mp = logging.getLogger(f'DL_{mp_idx}')
    
    def _save_thumbnail(arr, path):
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(arr)
        ax.axis('off')
        fig.savefig(path,bbox_inches='tight',pad_inches=0)
        plt.close()
    
    TLs = {}
    
    for idx, pt in pts.iterrows():
        

        
        # make the path
        if not os.path.exists(os.path.join(CONFIG['DATA_ROOT'], version,str(idx))):
            os.makedirs(os.path.join(CONFIG['DATA_ROOT'], version,str(idx)))
            
            
        # get the tile
        tile = raster_client.dltile_from_latlon(pt['lat'],pt['lon'], CONFIG['resolution'], CONFIG['patch_size'],0)
        #print (tile.geometry)
        
        #print (S2_min_date, S2_max_date, S1_min_date, S1_max_date)
        try:
            try:
                # get the arrays
                S2_arr, S2_meta = raster_client.ndarray(pt['DL_S2'], 
                                                    bands=CONFIG['DL']['S2_bands'], 
                                                    scales = [(0,10000,0,10000)]*(len(CONFIG['DL']['S2_bands'])-1) + [(0,1,0,1)],
                                                    data_type='Float32',
                                                    dltile=tile.properties.key, 
                                                    processing_level='surface'
                                                    )


            except:
                # search the dl catalog
                S2_min_date = dt.strptime(pt['DL_S2'].split(':')[2][0:10],'%Y-%m-%d') - timedelta(days=1)
                S2_max_date = S2_min_date + timedelta(days=2)

                S2_scenes, S2_ctx = dl.scenes.search(
                                    aoi=tile.geometry,
                                    products='sentinel-2:L1C',
                                    start_datetime=S2_min_date,
                                    end_datetime=S2_max_date
                                )

                _ids = [s._dict()['properties']['id'] for s in S2_scenes]

                print ('S2 backup search',pt['DL_S2'],[(_id, fuzz.ratio(_id,pt['DL_S2'])) for _id in _ids])
                _id = max(_ids, key=lambda _id: fuzz.ratio(_id,pt['DL_S2']))

                S2_arr, S2_meta = raster_client.ndarray(_id, 
                                                    bands=CONFIG['DL']['S2_bands'], 
                                                    scales = [(0,10000,0,10000)]*(len(CONFIG['DL']['S2_bands'])-1) + [(0,1,0,1)],
                                                    data_type='Float32',
                                                    dltile=tile.properties.key, 
                                                    processing_level='surface'
                                                    )



            try:
                S1_arr, S1_meta = raster_client.ndarray(pt['DL_S1'], 
                                            bands=CONFIG['DL']['S1_bands'], 
                                            scales = [(0,255,0,255)]*len(CONFIG['DL']['S1_bands']),
                                            data_type='Float32',
                                            dltile=tile.properties.key, 
                                            )

            except:
                # search the dl catalog
                S1_min_date = dt.strptime(pt['DL_S1'].split('_')[1],'%Y-%m-%d')- timedelta(days=1)
                S1_max_date = S2_min_date + timedelta(days=2)

                S1_scenes, S1_ctx = dl.scenes.search(
                                        aoi=tile.geometry,
                                        products='sentinel-1:GRD',
                                        start_datetime=S1_min_date,
                                        end_datetime=S1_max_date
                                    )

                _ids = [s._dict()['properties']['id'] for s in S1_scenes]

                print ('S1 backup search', pt['DL_S1'],[(_id, fuzz.ratio(_id,pt['DL_S1'])) for _id in _ids])
                _id = max(_ids, key=lambda _id: fuzz.ratio(_id,pt['DL_S1']))

                S1_arr, S1_meta = raster_client.ndarray(_id, 
                                            bands=CONFIG['DL']['S1_bands'], 
                                            scales = [(0,255,0,255)]*len(CONFIG['DL']['S1_bands']),
                                            data_type='Float32',
                                            dltile=tile.properties.key, 
                                            )



            name_root = os.path.join(os.getcwd(),'data',version,str(idx),'_'.join([str(idx),'DL',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

            np.savez(name_root+'_S2arr.npz', arr=S2_arr.astype(np.float32))
            json.dump(S2_meta, open(name_root+'_S2meta.json','w'))
            np.savez(name_root+'_S1arr.npz', arr=S1_arr.astype(np.float32))
            json.dump(S1_meta, open(name_root+'_S1meta.json','w'))
            json.dump(tile, open(name_root+'_tile.json','w'))



            _save_thumbnail((S2_arr[:,:,(3,2,1)].astype(np.float32)/10000).clip(0,1), name_root+'_S2thumb.png')
            _save_thumbnail((np.stack([S1_arr[:,:,0],np.zeros(S1_arr.shape[0:2]),S1_arr[:,:,1]])/255*2.5).clip(0,1).transpose([1,2,0]), name_root+'_S1thumb.png')

            print (f'done pt {idx}, S2_min: {S2_arr.min()}, S2_max: {S2_arr.max()}, S1_min: {S1_arr.min()}, S1_max: {S1_arr.max()}, {pt["DL_S2"]},{pt["lon"]},{pt["lat"]}')
        except Exception as e:
            print ('Error!')
            print (e)

        
        TLs[idx] = dict(tile)
    
    return TLs, ii_ps
    

                
    
    

def GEE_downloader(version, pts, ii_ps, CONFIG, mp_idx, TLs):

    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    KEY = os.path.join(os.getcwd(),'ee-deepsentinel.json')

    credentials = service_account.Credentials.from_service_account_file(KEY)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

    session = AuthorizedSession(scoped_credentials)
        
    logger_mp = logging.getLogger(f'GEE_{mp_idx}')
    
    PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    
    def _save_thumbnail(arr, path):
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(arr)
        ax.axis('off')
        fig.savefig(path,bbox_inches='tight',pad_inches=0)
        plt.close()
        
    def _get_GEE_arr(session, name, bands, x_off, y_off, patch_size, crs_code):
        url = 'https://earthengine.googleapis.com/v1alpha/{}:getPixels'.format(name)
        body = json.dumps({
            'fileFormat': 'NPY',
            'bandIds': bands,
            'grid': {
                'affineTransform': {
                    'scaleX': 10,
                    'scaleY': -10,
                    'translateX': x_off,
                    'translateY': y_off,
                },
                'dimensions': {'width': patch_size, 'height': patch_size}, #
                'crsCode': crs_code
            },
        })
        
        pixels_response = session.post(url, body)
        pixels_content = pixels_response.content
        
        # print (pixels_content)

        arr =  np.load(io.BytesIO(pixels_content))
        
        return np.dstack([arr[el] for el in arr.dtype.names]).astype(np.float32)
        
    for idx, pt in pts.iterrows():
        
        try:
        
            # make the path
            if not os.path.exists(os.path.join(CONFIG['DATA_ROOT'], version,str(idx))):
                os.makedirs(os.path.join(CONFIG['DATA_ROOT'], version,str(idx)))


            tl_avail = str(idx) in TLs.keys() #) and np.prod([kk in TLs[str(idx)].keys() for kk in ['utm_zone','x_off','y_off']])

            #print (idx, tl_avail, pt['lon'],pt['lat'])

            # get UTM zone
            if not tl_avail:
                utm_zone = get_utm_zone(pt['lat'],pt['lon'])

                pt_wgs = geometry.Point(pt['lon'],pt['lat'])

                # reprojection functions
                proj_utm = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')
                reproj_wgs_utm = partial(pyproj.transform, PROJ_WGS, proj_utm)
                reproj_utm_wgs = partial(pyproj.transform, proj_utm, PROJ_WGS)

                # get utm pt
                pt_utm = ops.transform(reproj_wgs_utm, pt_wgs)

                # get utm bbox
                bbox_utm = geometry.box(
                    pt_utm.x-(CONFIG['patch_size']*CONFIG['resolution'])/2, 
                    pt_utm.y-(CONFIG['patch_size']*CONFIG['resolution'])/2, 
                    pt_utm.x+(CONFIG['patch_size']*CONFIG['resolution'])/2, 
                    pt_utm.y+(CONFIG['patch_size']*CONFIG['resolution'])/2)

                # reproj to wgs
                bbox_wgs = ops.transform(reproj_utm_wgs, bbox_utm)

                x_off, y_off = bbox_utm.bounds[0], bbox_utm.bounds[1]
                
                if pt['lat']>0:
                    UTM_EPSG = f'EPSG:{str(326)+str(utm_zone)}'
                else:
                    UTM_EPSG = f'EPSG:{str(327)+str(utm_zone)}'

            else:
                utm_zone = TLs[str(idx)]['properties']['zone']
                x_off, y_off = TLs[str(idx)]['properties']['geotrans'][0], TLs[str(idx)]['properties']['geotrans'][3]
                UTM_EPSG = TLs[str(idx)]['properties']['cs_code']


            #print ('utm_zone',utm_zone,TLs[str(idx)]['properties']['zone'])
            #print ('x_off',x_off, TLs[str(idx)]['properties']['geotrans'][0])
            #print ('y_off',y_off,TLs[str(idx)]['properties']['geotrans'][3])
            #exit()

            


            S2_arr = _get_GEE_arr(
                session=session, 
                name=pt['GEE_S2'], 
                bands=CONFIG['GEE']['S2_bands'], 
                x_off=x_off, 
                y_off=y_off, 
                patch_size=CONFIG['patch_size'],
                crs_code=UTM_EPSG
            )

            S1_arr = _get_GEE_arr(
                session=session, 
                name=pt['GEE_S1'], 
                bands=CONFIG['GEE']['S1_bands'], 
                x_off=x_off, 
                y_off=y_off, 
                patch_size=CONFIG['patch_size'],
                crs_code=UTM_EPSG
            )

            if tl_avail:
                tile = TLs[str(idx)]
            else:
                tile = None

            GEE_meta = {
                'tile':tile,
                'S2_name':pt['GEE_S2'],
                'S1_name':pt['GEE_S1'],
                'S2_bands':CONFIG['GEE']['S2_bands'],
                'S1_bands':CONFIG['GEE']['S1_bands'],
                'x_off':x_off, 
                'y_off':y_off, 
                'patch_size':CONFIG['patch_size'], 
                'crs_code':UTM_EPSG,
            }


            name_root = os.path.join(CONFIG['DATA_ROOT'],version,str(idx),'_'.join([str(idx),'GEE',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

            np.savez(name_root+'_S2arr.npz', arr=S2_arr.astype(np.float32))
            np.savez(name_root+'_S1arr.npz', arr=S1_arr.astype(np.float32))
            json.dump(GEE_meta, open(name_root+'_GEE_meta.json','w'))

            _save_thumbnail((S2_arr[:,:,(3,2,1)]/10000).clip(0,1), name_root+'_S2thumb.png')
            _save_thumbnail((((np.stack([S1_arr[:,:,0],np.zeros(S1_arr.shape[0:2]),S1_arr[:,:,1]])).transpose([1,2,0])+np.abs(S1_arr.min()))/(S1_arr.max()-S1_arr.min())).clip(0,1), name_root+'_S1thumb.png')
            print (f'Done pt {idx}, S2_min: {S2_arr.min()}, S2_max: {S2_arr.max()}, S1_min: {S1_arr.min()}, S1_max: {S1_arr.max()}')
        except Exception as e:
            print (f'Error pt {idx}: {e}')
        
    return True


class SampleDownloader:
    
    
    def __init__(self, version, use_dl, use_gee, multiprocess=False):

        # load config, credentials
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)
        
        self.version = version
        
        self.use_dl = use_dl
        
        self.use_gee = use_gee
        
        # load the point df
        self.pts = pd.read_parquet(os.path.join(self.CONFIG['DATA_ROOT'], 'pts', self.version+'.parquet'))
        
        # make directory
        if not os.path.exists(os.path.join(self.CONFIG['DATA_ROOT'],self.version)):
            os.makedirs(os.path.join(self.CONFIG['DATA_ROOT'],self.version))
        

            
            
    def download_samples_DL(self):
        
        args = []
        step = (len(self.pts)//self.CONFIG['N_workers'])+1

        for ii_w in range(self.CONFIG['N_workers']):
            args.append(
                (self.version,
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:],
                 self.pts.iloc[ii_w*step:(ii_w+1)*step,:].index.values,
                 self.CONFIG,
                 ii_w
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
        
        GEE_downloader(self.version, self.pts, self.pts.index.values, self.CONFIG, 0, TLs)
        

        
        
if __name__=="__main__":
    downloader=SampleDownloader(version='v_null_2', use_dl=True, use_gee=False)
    downloader.download_samples_DL()
    downloader.download_samples_GEE()