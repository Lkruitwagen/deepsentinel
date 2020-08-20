import os, json, yaml, zipfile, geojson, urllib, re, io
import pyproj
from functools import partial
from shapely import geometry, ops
from datetime import datetime as dt 
from datetime import timedelta
import numpy as np
import pandas as pd
from importlib import import_module

import matplotlib.pyplot as plt


from deepsentinel.utils.geoutils import *


class FOSSDownloader():

    def __init__(self):
        from sentinelsat.sentinel import SentinelAPI
        # load config, credentials

        self.cred = json.load(open(os.path.join(os.getcwd(),'credentials.json'),'r'))
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        # 
        self.api = SentinelAPI(self.cred['scihub']['U'], self.cred['scihub']['P'], 'https://scihub.copernicus.eu/dhus')

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)


    def download_pts(self, pts):

        for pt in pts:
            print ('Running pt',pt)

            # get product: JSON ['S2','S1']
            products = self._get_products(pt)

            if products:
                self._get_sample(pt, products)


    def _get_products(self, pt):
        
        # pt lon/lat to UTM square
        utm_zone = get_utm_zone(pt.y, pt.x)
        self.proj_utm = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')

        # reprojection functions
        reproj_wgs_utm = partial(pyproj.transform, self.proj_wgs, self.proj_utm)
        reproj_utm_wgs = partial(pyproj.transform, self.proj_utm, self.proj_wgs)

        # get utm pt
        pt_utm = ops.transform(reproj_wgs_utm, pt)

        # get utm bbox
        bbox_utm = geometry.box(
            pt_utm.x-(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
            pt_utm.y-(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
            pt_utm.x+(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
            pt_utm.y+(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2)

        # reproj to wgs
        bbox_wgs = ops.transform(reproj_utm_wgs, bbox_utm)

        # Get query dates. Randomise over a year for seasonality
        Q_date_min = self.min_date+timedelta(days=np.random.choice(365-self.CONFIG['dt_lookup']))
        Q_date_max = Q_date_min + timedelta(days=self.CONFIG['dt_lookup'])

        # get S2 products
        S2_products = self.api.query(bbox_wgs.wkt,
                             date = (Q_date_min, Q_date_max),
                             platformname = 'Sentinel-2',
                             cloudcoverpercentage = (0, 20))

        S1_products = self.api.query(bbox_wgs.wkt,
                     date = (Q_date_min, Q_date_max),
                     platformname = 'Sentinel-1')

        # Cast to dataframes and get L2A only
        S2_df = pd.DataFrame.from_dict(S2_products, orient='index')
        S1_df = pd.DataFrame.from_dict(S1_products, orient='index')
        S2_df = S2_df.loc[S2_df['processinglevel']=='Level-2A']
        S1_df = S1_df.loc[S1_df['producttype']=='GRD']

        # get matches (just apply)
        S2_df['matches'] = S2_df.beginposition.apply(lambda ts: S1_df.loc[(S1_df['beginposition']<=(ts+timedelta(days=self.CONFIG['day_offset']))).values & (S1_df['beginposition']>=(ts-timedelta(days=self.CONFIG['day_offset']))).values].index.values.tolist())
        S2_df['selfmatches'] = S2_df.beginposition.apply(lambda ts: S2_df.loc[(S2_df['beginposition']<=(ts+timedelta(hours=self.CONFIG['selfmatch_offset']))).values & (S2_df['beginposition']>=(ts-timedelta(hours=self.CONFIG['selfmatch_offset']))).values].index.values.tolist())



        # get 0th 
        S2_root = S2_df.loc[S2_df['beginposition']>=Q_date_min+timedelta(days=self.CONFIG['day_offset'])].iloc[0]

        S2_select = S2_df.loc[S2_df.index.isin(S2_root['selfmatches']),:]
        S1_select = S1_df.loc[S1_df.index.isin(S2_root['matches']),:]


        if S2_df.matches.str.len().sum()==0:
            return None
        else:
            return {'S2':S2_select, 'S1':S1_select}

    def _maybe_download(self, rec, constellation):

        if not os.path.exists(os.path.join(os.getcwd(),'data','raw',rec.title+'.SAFE')):

            print (os.path.join(os.getcwd(),'data','raw',rec.title+'.SAFE'), 'not found, downloading...')

            self.api.download(rec.name, directory_path=os.path.join(os.getcwd(),'data','raw'))

            with zipfile.ZipFile(os.path.join(os.getcwd(),'data','raw',rec.title+'.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(os.getcwd(),'data','raw',rec.title+'.SAFE'))

            rec.to_json(os.path.join(os.getcwd(),'data','raw',rec.title+'.json'))

        else:
            print ('Found ', rec.title)


    def _get_sample(self, pt, products):
        
        if self.CONFIG['mosaic']:
            raise NotImplementedError
        
        else:
            print (products['S2'].iloc[0].name)
            print (products['S2'])


            self._maybe_download(products['S2'].iloc[0],'S2')
            self._maybe_download(products['S1'].iloc[0],'S1')

            # gdalwarp not good enough ortho. Try ESA SNAP






class DLDownloader:

    def __init__(self):
        import descarteslabs as dl
        # load config, credentials
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)

        self.raster_client = dl.Raster()
        self.dl = dl



    def download_pts(self, pts):

        for ii_p, pt in enumerate(pts):
            print ('Running pt',pt)

            # get product: JSON ['S2','S1']
            products = self._get_products(pt)

            if products:
                self._get_sample(pt, products, ii_p)


    def _get_products(self, pt):

        # get dltile
        self.tile = self.raster_client.dltile_from_latlon(pt.y, pt.x, self.CONFIG['resolution'], self.CONFIG['patch_size'], 0)
        

        # Get query dates. Randomise over a year for seasonality
        Q_date_min = self.min_date+timedelta(days=np.random.choice(365-self.CONFIG['dt_lookup']))
        Q_date_max = Q_date_min + timedelta(days=self.CONFIG['dt_lookup'])


        # get S2 products
        S2_scenes, self.S2_ctx = self.dl.scenes.search(
                                aoi=self.tile.geometry,
                                products='sentinel-2:L1C',
                                start_datetime=Q_date_min,
                                end_datetime=Q_date_max
                            )

        S1_scenes, self.S1_ctx = self.dl.scenes.search(
                                aoi=self.tile.geometry,
                                products='sentinel-1:GRD',
                                start_datetime=Q_date_min,
                                end_datetime=Q_date_max
                            )

        # Cast to dataframes
        S2_df = pd.DataFrame.from_records([s._dict()['properties'] for s in S2_scenes])
        S1_df = pd.DataFrame.from_records([s._dict()['properties'] for s in S1_scenes])
        S2_df['acquired'] = pd.to_datetime(S2_df['acquired']).dt.tz_localize(None)
        S1_df['acquired'] = pd.to_datetime(S1_df['acquired']).dt.tz_localize(None)

        # get matches (just apply)
        S2_df['matches'] = S2_df.acquired.apply(lambda ts: S1_df.loc[(S1_df['acquired']<=(ts+timedelta(days=self.CONFIG['day_offset']))).values & (S1_df['acquired']>=(ts-timedelta(days=self.CONFIG['day_offset']))).values,'id'].values.tolist())
        S2_df['selfmatches'] = S2_df.acquired.apply(lambda ts: S2_df.loc[(S2_df['acquired']<=(ts+timedelta(hours=self.CONFIG['selfmatch_offset']))).values & (S2_df['acquired']>=(ts-timedelta(hours=self.CONFIG['selfmatch_offset']))).values,'id'].values.tolist())



        # get 0th 
        S2_root = S2_df.loc[S2_df['acquired']>=Q_date_min+timedelta(days=self.CONFIG['day_offset'])].iloc[0]

        S2_select = S2_df.loc[S2_df['id'].isin(S2_root['selfmatches']),:]
        S1_select = S1_df.loc[S1_df['id'].isin(S2_root['matches']),:]


        if S2_df.matches.str.len().sum()==0:
            return None
        else:
            return {'S2':S2_select, 'S1':S1_select}




    def _get_sample(self, pt, products,ii_p):

        if not os.path.exists(os.path.join(os.getcwd(),'data','DL',str(ii_p))):
            os.makedirs(os.path.join(os.getcwd(),'data','DL',str(ii_p)))



        def save_thumbnail(arr, path):
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.imshow(arr)
            ax.axis('off')
            fig.savefig(path,bbox_inches='tight',pad_inches=0)
        
        if self.CONFIG['mosaic']:
            raise NotImplementedError
        
        else:

            # S2

            S2_arr, S2_meta = self.raster_client.ndarray(products['S2'].iloc[0]['id'], 
                                        bands=self.CONFIG['DL']['S2_bands'], 
                                        scales = [(0,10000,0,255)]*(len(self.CONFIG['DL']['S2_bands'])-1) + [(0,1,0,1)],
                                        data_type='Byte',
                                        dltile=self.tile.properties.key, 
                                        processing_level='surface'
                                        )

            S1_arr, S1_meta = self.raster_client.ndarray(products['S1'].iloc[0]['id'], 
                                        bands=self.CONFIG['DL']['S1_bands'], 
                                        scales = [(0,255,0,255)]*len(self.CONFIG['DL']['S1_bands']),
                                        data_type='Byte',
                                        dltile=self.tile.properties.key, 
                                        )

            name_root = '_'.join([str(ii_p),str(products['S2'].iloc[0]['acquired'])[0:10], str(pt.x), str(pt.y)])

            np.savez(os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S2arr.npz'), arr=S2_arr)
            json.dump(S2_meta, open(os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S2meta.json'),'w'))
            np.savez(os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S1arr.npz'), arr=S1_arr)
            json.dump(S1_meta, open(os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S1meta.json'),'w'))


            save_thumbnail((S2_arr[:,:,(1,2,3)]/255*2.5).clip(0,1), os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S2thumb.png'))
            save_thumbnail((np.stack([S1_arr[:,:,0], np.zeros(S1_arr.shape[0:2]),S1_arr[:,:,1]])/255*2.5).clip(0,1).transpose([1,2,0]), os.path.join(os.getcwd(),'data','DL',str(ii_p),name_root+'_S1thumb.png'))



class GEEDownloader:

    def __init__(self):
        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account
        # load config, credentials
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        KEY = os.path.join(os.getcwd(),'ee-deepsentinel.json')

        credentials = service_account.Credentials.from_service_account_file(KEY)
        scoped_credentials = credentials.with_scopes(
            ['https://www.googleapis.com/auth/cloud-platform'])

        self.session = AuthorizedSession(scoped_credentials)

        self.proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

        self.max_date = dt.strptime(self.CONFIG['max_date'],'%Y-%m-%d')
        self.min_date = self.max_date - timedelta(days=365)



    def download_pts(self, pts):

        for ii_p, pt in enumerate(pts):
            print ('Running pt',pt)

            # get product: JSON ['S2','S1']
            products = self._get_products(pt)

            if products:
                self._get_sample(pt, products, ii_p)


    def _get_products(self, pt):



        # Get query dates. Randomise over a year for seasonality
        Q_date_min = self.min_date+timedelta(days=np.random.choice(365-self.CONFIG['dt_lookup']))
        Q_date_max = Q_date_min + timedelta(days=self.CONFIG['dt_lookup'])


        S1_name = 'projects/earthengine-public/assets/COPERNICUS/S1_GRD'
        S2_name = 'projects/earthengine-public/assets/COPERNICUS/S2'

        # get S2 products
        url = 'https://earthengine.googleapis.com/v1alpha/{}:listImages?{}'.format(
                  S2_name, urllib.parse.urlencode({
                    'startTime':Q_date_min.isoformat()+'.000Z',
                    'endTime': Q_date_max.isoformat()+'.000Z',
                    'region': '{"type":"Point", "coordinates":' + str([pt.x,pt.y]) + '}'}))
        response = self.session.get(url)
        content = json.loads(response.content)

        S2_df = pd.DataFrame.from_records(content['images'])


        # get S1 products
        url = 'https://earthengine.googleapis.com/v1alpha/{}:listImages?{}'.format(
                  S1_name, urllib.parse.urlencode({
                    'startTime':Q_date_min.isoformat()+'.000Z',
                    'endTime': Q_date_max.isoformat()+'.000Z',
                    'region': '{"type":"Point", "coordinates":' + str([pt.x,pt.y]) + '}'}))
        response = self.session.get(url)
        content = json.loads(response.content)

        S1_df = pd.DataFrame.from_records(content['images'])


        # Cast to dataframes
        S2_df['startTime'] = pd.to_datetime(S2_df['startTime']).dt.tz_localize(None)
        S1_df['startTime'] = pd.to_datetime(S1_df['startTime']).dt.tz_localize(None)

        # get matches (just apply)
        S2_df['matches'] = S2_df.startTime.apply(lambda ts: S1_df.loc[(S1_df['startTime']<=(ts+timedelta(days=self.CONFIG['day_offset']))).values & (S1_df['startTime']>=(ts-timedelta(days=self.CONFIG['day_offset']))).values,'id'].values.tolist())
        S2_df['selfmatches'] = S2_df.startTime.apply(lambda ts: S2_df.loc[(S2_df['startTime']<=(ts+timedelta(hours=self.CONFIG['selfmatch_offset']))).values & (S2_df['startTime']>=(ts-timedelta(hours=self.CONFIG['selfmatch_offset']))).values,'id'].values.tolist())



        # get 0th 
        S2_root = S2_df.loc[S2_df['startTime']>=Q_date_min+timedelta(days=self.CONFIG['day_offset'])].iloc[0]

        S2_select = S2_df.loc[S2_df['id'].isin(S2_root['selfmatches']),:]
        S1_select = S1_df.loc[S1_df['id'].isin(S2_root['matches']),:]


        if S2_df.matches.str.len().sum()==0:
            return None
        else:
            return {'S2':S2_select, 'S1':S1_select}




    def _get_sample(self, pt, products,ii_p):

        if not os.path.exists(os.path.join(os.getcwd(),'data','GEE',str(ii_p))):
            os.makedirs(os.path.join(os.getcwd(),'data','GEE',str(ii_p)))



        def save_thumbnail(arr, path):
            fig, ax = plt.subplots(1,1,figsize=(6,6))
            ax.imshow(arr)
            ax.axis('off')
            fig.savefig(path,bbox_inches='tight',pad_inches=0)

        def norm(arr):
            arr = arr+np.abs(arr.min())
            arr = arr/arr.max()
            return arr
        
        if self.CONFIG['mosaic']:
            raise NotImplementedError
        
        else:

            # get the UTM zone
            utm_zone = re.findall(r'\d+',products['S2'].iloc[0].properties['MGRS_TILE'])[0]
            self.proj_utm = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')

            # reprojection functions
            reproj_wgs_utm = partial(pyproj.transform, self.proj_wgs, self.proj_utm)
            reproj_utm_wgs = partial(pyproj.transform, self.proj_utm, self.proj_wgs)

            # get utm pt
            pt_utm = ops.transform(reproj_wgs_utm, pt)

            # get utm bbox
            bbox_utm = geometry.box(
                pt_utm.x-(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
                pt_utm.y-(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
                pt_utm.x+(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2, 
                pt_utm.y+(self.CONFIG['patch_size']*self.CONFIG['resolution'])/2)

            # reproj to wgs
            bbox_wgs = ops.transform(reproj_utm_wgs, bbox_utm)

            url = 'https://earthengine.googleapis.com/v1alpha/{}:getPixels'.format(products['S2'].iloc[0]['name'])
            body = json.dumps({
                'fileFormat': 'NPY',
                'bandIds': self.CONFIG['GEE']['S2_bands'],
                'grid': {
                    'affineTransform': {
                        'scaleX': 10,
                        'scaleY': -10,
                        'translateX': int(bbox_utm.bounds[0]),#x_off,
                        'translateY': int(bbox_utm.bounds[1]),#y_off,
                    },
                    'dimensions': {'width': 256, 'height': 256},
                },
            })

            pixels_response = self.session.post(url, body)
            pixels_content = pixels_response.content
            #print (pixels_content)
            S2_arr =  np.load(io.BytesIO(pixels_content))
            S2_arr = np.dstack([S2_arr[el] for el in S2_arr.dtype.names])

            if pt.y>0:
                UTM_EPSG = f'EPSG:{str(326)+utm_zone}'
            else:
                UTM_EPSG = f'EPSG:{str(327)+utm_zone}'

            print ('UTM EPSG', UTM_EPSG)


            url = 'https://earthengine.googleapis.com/v1alpha/{}:getPixels'.format(products['S1'].iloc[0]['name'])
            body = json.dumps({
                'fileFormat': 'NPY',
                'bandIds': self.CONFIG['GEE']['S1_bands'],
                'grid': {
                    'affineTransform': {
                        'scaleX': 10,
                        'scaleY': -10,
                        'translateX': int(bbox_utm.bounds[0]),#x_off,
                        'translateY': int(bbox_utm.bounds[1]),#y_off,
                    },
                    'dimensions': {'width': 256, 'height': 256},
                    'crsCode': UTM_EPSG
                },
            })

            pixels_response = self.session.post(url, body)
            pixels_content = pixels_response.content
            S1_arr =  np.load(io.BytesIO(pixels_content))
            S1_arr = np.dstack([S1_arr[el] for el in S1_arr.dtype.names])


            name_root = '_'.join([str(ii_p),str(products['S2'].iloc[0]['startTime'])[0:10], str(pt.x), str(pt.y)])

            np.savez(os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S2arr.npz'), arr=S2_arr)
            json.dump(products['S2'].iloc[0].to_json(), open(os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S2meta.json'),'w'))
            np.savez(os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S1arr.npz'), arr=S1_arr)
            json.dump(products['S2'].iloc[0].to_json(), open(os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S1meta.json'),'w'))


            save_thumbnail((S2_arr[:,:,(1,2,3)]/10000*2.5).clip(0,1), os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S2thumb.png'))
            save_thumbnail((np.stack([norm(S1_arr[:,:,0]),np.zeros(S1_arr.shape[0:2]),norm(S1_arr[:,:,1])])).clip(0,1).transpose([1,2,0]), os.path.join(os.getcwd(),'data','GEE',str(ii_p),name_root+'_S1thumb.png'))



if __name__=="__main__":
    downloader = GEEDownloader()

    pts =  [geometry.Point(-1.254269, 51.750400),
            geometry.Point(4.609613597470932,52.39142947728902),
            geometry.Point(-74.42971915221149,46.081681155014884)]

    downloader.download_pts(pts)