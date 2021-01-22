"""
A class to generate a set of points for the imagery downloader.
"""
import os, yaml, time, random, json, glob, logging, re
from tqdm import tqdm
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
import geopandas as gpd
import numpy as np
import pygeos
from shapely import wkt, geometry
from area import area
from google.cloud import storage

gpd.options.use_pygeos = True

from deepsentinel.utils.download_catalog import async_product_caller, async_product_worker
from deepsentinel.utils.geoutils import pt2bbox_wgs
from deepsentinel.utils.utils import async_load_parquets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointGenerator:

    def __init__(self, iso_geographies=None, conf=None):
        
        if not conf:
            self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)
        else:
            self.CONFIG = yaml.load(open(conf,'r'),Loader=yaml.SafeLoader)
        
        self.sentinelsat_auth = json.load(open(self.CONFIG['scihub_auth'],'r'))['scihub']

        countries = gpd.read_file(os.path.join(self.CONFIG['NE_ROOT'],'ne_10m_countries.gpkg'))
        countries = countries[~countries['geometry'].isna()]
        
        if iso_geographies:
            countries = countries[countries['ISO_A2'].isin(iso_geographies)]
        
        self.tree = pygeos.STRtree([pygeos.io.from_shapely(subshp) for subshp in list(countries['geometry'].unary_union)])
        
        self.S2_tiles = gpd.read_file(os.path.join(self.CONFIG['NE_ROOT'],'S2_utm.gpkg'))
        
        
        # Get datastrip_ids after Nov-2019 from GCP buckets
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.CONFIG['gcp_credentials_path']
        self.client = storage.Client()
        self.bucketname = 'gcp-public-data-sentinel-2'
    
    
    def check_catalog(self):
        
        data_files = sorted(glob.glob(os.path.join(self.CONFIG['CATALOG_ROOT'],'*.parquet')))
        # platform, product, date
        data_records = [{
            'platform':os.path.split(f)[1].split('_')[0],
            'product':os.path.split(f)[1].split('_')[1],
            'date':os.path.split(f)[1].split('_')[-1][0:10],
            'f':f
        } 
        for f in data_files]
        
        self.catalog = pd.DataFrame.from_records(data_records)
        self.catalog['date'] = pd.to_datetime(self.catalog['date'])
        
    def get_missing_records(self,start_date, N_orbits):
        
        all_dates = np.array([start_date+timedelta(days=ii) for ii in range(N_orbits*self.CONFIG['orbit_period'])])
        missing_records = []
        
        for platform, product in [('Sentinel-2','S2MSI2A'),('Sentinel-2','S2MSI1C'),('Sentinel-1','GRD')]:
            missing_dates = all_dates[~pd.Series(all_dates).isin(self.catalog.loc[(self.catalog['platform']==platform)&(self.catalog['product']==product),'date']).values]
            missing_records += [(platform, product, dd, self.sentinelsat_auth, self.CONFIG['CATALOG_ROOT']) for dd in missing_dates]
            
        return missing_records
        

    def load_catalog(self, min_date):
        S1_df = async_load_parquets(self.catalog.loc[(self.catalog['date']>=min_date) & (self.catalog['date']<(min_date+timedelta(days=self.CONFIG['orbit_period']))) & (self.catalog['platform']=='Sentinel-1'),'f'].values.tolist(), self.CONFIG['N_workers'])
        S2_L1C_df = async_load_parquets(self.catalog.loc[(self.catalog['date']>=min_date) & (self.catalog['date']<(min_date+timedelta(days=self.CONFIG['orbit_period']))) & (self.catalog['product']=='S2MSI1C'),'f'].values.tolist(), self.CONFIG['N_workers'])
        S2_L2A_df = async_load_parquets(self.catalog.loc[(self.catalog['date']>=min_date) & (self.catalog['date']<(min_date+timedelta(days=self.CONFIG['orbit_period']))) & (self.catalog['product']=='S2MSI2A'),'f'].values.tolist(), self.CONFIG['N_workers'])
        
        return S1_df, S2_L1C_df, S2_L2A_df
        
        

    def main_generator(self,start_date, N_orbits, N_points, name):


        ### check and maybe download catalog  
        logger.info('Loading catalog.')
        self.check_catalog()
        
        missing_records = self.get_missing_records(start_date, N_orbits)
        if len(missing_records)>0:
            logger.info(f'Found missing records: {len(missing_records)}, calling async catalog download with {self.CONFIG["N_workers"]} workers')
            
            async_product_caller(missing_records, self.CONFIG['N_workers'])
        
            self.check_catalog()
        else:
            logger.info('Found all needed records.')
            
            
        all_pts = []
        

        logger.info(f'Generating points for N_orbits: {N_orbits}')

        for orbit in range(N_orbits):
            
            tic = time.time()
            
            
            S1_df, S2_L1C_df, S2_L2A_df = self.load_catalog(start_date + timedelta(days=self.CONFIG['orbit_period']*orbit))
            
            # filter only desired polarisation for S1
            S1_df = S1_df[S1_df['polarisationmode']=='VV VH']
            
            S2_L1C_df['beginposition'] = pd.to_datetime(S2_L1C_df['beginposition'])
            S2_L2A_df['beginposition'] = pd.to_datetime(S2_L2A_df['beginposition'])
            S1_df['beginposition'] = pd.to_datetime(S1_df['beginposition'])
            S1_df['endposition'] = pd.to_datetime(S1_df['endposition'])
            
            # only retain S2 records where there is both L1C and L2A
            S2_L2A_df = S2_L2A_df[S2_L2A_df['level1cpdiidentifier'].isin(S2_L1C_df['level1cpdiidentifier'])]
            S2_L2A_df = S2_L2A_df.set_index('level1cpdiidentifier') # looks like it's just used to match them.
            S2_L1C_df = S2_L1C_df.set_index('level1cpdiidentifier')
            
            #print ('len S2', len(S2_L2A_df))
            
            # get S2_L2A_df geometry intersection
            S2_L2A_df['utm_tile'] = S2_L2A_df['title'].str.split('_').str[5].str[1:]
            S2_L2A_df = pd.merge(S2_L2A_df.reset_index(), self.S2_tiles[['Name','geometry']], how='left',left_on='utm_tile',right_on='Name')
            S2_L2A_df = S2_L2A_df.set_index('level1cpdiidentifier')
            S2_L2A_df['intersection_geom'] = S2_L2A_df.apply(lambda row: row['geometry'].intersection(wkt.loads(row['footprint'])), axis=1)
            
            # get coverage
            S2_L2A_df['coverage'] = S2_L2A_df.apply(lambda row: area(geometry.mapping(row['intersection_geom']))/area(geometry.mapping(wkt.loads(row['footprint']))), axis=1)
            #print ('got coverage',time.time()-tic)
            #S2_L2A_df['naive_coverage'] = S2_L2A_df.apply(lambda row: row['intersection_geom'].area/wkt.loads(row['footprint']).area, axis=1)
            


            pts = pd.DataFrame(columns=['lon','lat','bbox_wgs','matches'])
            
            pbar = tqdm(total=N_points, desc=f'orbit: {orbit}, start_date: {start_date + timedelta(days=self.CONFIG["orbit_period"]*orbit)}', ncols=100)


            while len(pts)<N_points:

                # obtain points
                new_pts = self._sample_land_points(N_points-len(pts))
                #print ('got new pts',len(new_pts), time.time()-tic)

                new_pts['bbox_wgs'] = new_pts.apply(lambda pt: pt2bbox_wgs(pt, patch_size=self.CONFIG['patch_size'], resolution=self.CONFIG['resolution'], use_pygeos=True), axis=1)
                #print ('got bboxes', time.time()-tic)

                # add matches
                new_pts = self._get_matches(new_pts, S1_df, S2_L2A_df)
                #print ('got matches', time.time()-tic)

                pts = pts.append(new_pts)
                
                pbar.update(len(new_pts))
                
            pbar.close()

            ### interim inspection:
            #out_pts = pts
            #out_pts['bbox_wgs'] = out_pts['bbox_wgs'].apply(pygeos.io.to_wkt)
            #out_pts.to_parquet(f'./all_matches_{orbit}.parquet')

            # match pts to DL and GEE
            pts['matches'] = pts['matches'].apply(lambda el: el[np.random.choice(len(el))])
            pts['S1_rec'] = pts['matches'].apply(lambda match: S1_df.loc[match[1],:].to_dict())
            pts['S2_L2A_rec'] = pts['matches'].apply(lambda match: S2_L2A_df.loc[match[0],:].to_dict())
            pts['S2_L1C_rec'] = pts['matches'].apply(lambda match: S2_L1C_df.loc[match[0],:].to_dict())
            
            #pts['coverage'] = pts.apply(self._map_coverage, axis=1)
            #pts['coverage_naive'] = pts.apply(self._map_coverage_naive, axis=1)

            pts['DL_S1'] = pts.apply(self._map_DL_S1, axis=1)
            pts['DL_S2'] = pts.apply(self._map_DL_S2, axis=1)
            pts['GEE_S1'] =  pts['S1_rec'].apply(lambda el: 'projects/earthengine-public/assets/COPERNICUS/S1_GRD/'+el['title'])
            pts['GEE_S2'] = pts.apply(self._map_GEE_S2, axis=1)
            
            all_pts.append(pts)
            
        all_pts = pd.concat(all_pts)
        
        logger.info('Got all points. Final formatting.')
        
        # format for disk
        all_pts['bbox_wgs'] = all_pts['bbox_wgs'].apply(pygeos.io.to_wkt)
        for col in ['S1_rec', 'S2_L2A_rec','S2_L1C_rec']:
            all_pts[col] = all_pts[col].apply(lambda el: el['title'])
            
        all_pts['idx'] = range(len(all_pts))
        all_pts = all_pts.set_index('idx')
                
        logger.info(f'Generated {len(all_pts)} points. Writing to Parquet at {os.path.join(self.CONFIG["POINTS_ROOT"],name+".parquet")}')
        
        all_pts.to_parquet(os.path.join(self.CONFIG['POINTS_ROOT'],name+'.parquet'))
            
            
    def _map_coverage(self, el):
        utm_tile = el['S2_L1C_rec']['title'].split('_')[5][1:]
        
        geom1 = self.S2_tiles[self.S2_tiles.Name==utm_tile].iloc[0]['geometry']
        geom2 = wkt.loads(el['S2_L1C_rec']['footprint'])
        
        return area(geometry.mapping(geom2.intersection(geom1)))/area(geometry.mapping(geom1))
    
    def _map_coverage_naive(self, el):
        utm_tile = el['S2_L1C_rec']['title'].split('_')[5][1:]
        
        geom1 = self.S2_tiles[self.S2_tiles.Name==utm_tile].iloc[0]['geometry']
        geom2 = wkt.loads(el['S2_L1C_rec']['footprint'])
        
        return geom2.intersection(geom1).area/geom1.area
    
    def _map_DL_S1(self,el):
        base='sentinel-1:GRD:meta'
        mean_dt = el['S1_rec']['beginposition']+(el['S1_rec']['endposition']-el['S1_rec']['beginposition'])/2
        dd=mean_dt.isoformat()[0:10]
        meta=f'{el["S1_rec"]["relativeorbitnumber"]:03d}{el["S1_rec"]["orbitdirection"][0]}{mean_dt.minute:02d}{mean_dt.second:02d}'
        satellite = el['S1_rec']['title'].split('_')[0]
        
        return '_'.join([base,dd,meta,satellite])

    def _map_DL_S2(self,el):
        base='sentinel-2:L1C:'
        mean_dt = el['S2_L1C_rec']['beginposition']+(el['S2_L1C_rec']['endposition']-el['S2_L1C_rec']['beginposition'])/2
        dd=mean_dt.isoformat()[0:10]
        utm_tile = el['S2_L1C_rec']['title'].split('_')[5][1:]
        satellite = el['S2_L1C_rec']['title'].split('_')[0]
        
        if round(el['S2_L2A_rec']['coverage']*100)==100:
            coverage_str = '99'
        else:
            coverage_str = str(int(np.round(el['S2_L2A_rec']['coverage']*100)))
        
        return base+'_'.join([dd,utm_tile,coverage_str,satellite,'v1'])
    
    def _map_GEE_S2(self,el):
        base = 'projects/earthengine-public/assets/COPERNICUS/S2_SR/'
        dt0 = el['S2_L2A_rec']['s2datatakeid'].split('_')[1]
        utm_tile = 'T'+el['S2_L2A_rec']['title'].split('_')[5][1:]
        
        if dt.strptime(dt0,'%Y%m%dT%H%M%S')>dt(2019,11,1,0,0) or el['S2_L1C_rec']['datastripidentifier'].split('_')[-2][1:]==None:
            # changed the manifest - now need to get this some other way.
            # either download the whole index.csv.gz or use big query, etc.
            tt = re.search(r'(\d).',utm_tile).group()
            big_grid = utm_tile.replace(tt,'')[1:][0]
            little_grid = utm_tile.replace(tt,'')[1:][1:]
            
            path=os.path.join('L2','tiles',tt,big_grid,little_grid,el['S2_L2A_rec']['filename'],'DATASTRIP')
            
            granule_blobs = [blob.name for blob in self.client.list_blobs(self.bucketname,prefix=path)]
            
            select_blob = [b for b in granule_blobs if ((os.path.split(b)[-1][0:3]=='DS_') and (os.path.split(b)[-1][-9:]=='_$folder$'))][0]
            
            dt1 = select_blob.split('_')[-2][1:]
            
            
        else:
            #try:
            dt1 = el['S2_L1C_rec']['datastripidentifier'].split('_')[-2][1:]
            #except:
            #    print (el['S2_L1C_rec'])
            #    raise ValueError('a big bork')

        return base+'_'.join([dt0,dt1,utm_tile])

            
        
            

    def _sample_land_points(self,N_pts):

        pt_df = pd.DataFrame(columns=['lon','lat'], index=[])

        
        ii_p=0
        while len(pt_df)<N_pts:

            pts = np.random.rand(2*2*(N_pts-len(pt_df))).reshape((N_pts-len(pt_df))*2,2)
            #print ('pts shape',pts.shape)
            pts[:,0] = pts[:,0] * 360 - 180 # lon
            pts[:,1] = pts[:,1] * (80 + 70) - 70 # lat between -70 and 80

            pts_pygeos = pygeos.points(pts)

            Q = self.tree.query_bulk(pts_pygeos, predicate='within').T[:,0]
            #print ('Q-shape',Q.shape)


            pt_df = pt_df.append(pd.DataFrame(pts[Q], columns=['lon','lat']))
            #print ('len pt_df', len(pt_df), 'iter_toc',time.time()-tic)
            ii_p+=1

        pt_df = pt_df.iloc[0:N_pts]
        
        # reset the index
        pt_df.index = range(len(pt_df))

        return pt_df


    def _get_matches(self, pts, S1_df, S2_df):

        def apply_matches(pt):
            S1_slice = S1_df.iloc[Q_S1.loc[Q_S1['pt_idx']==pt.name,'S1_idx'],:]
            S2_slice = S2_df.iloc[Q_S2.loc[Q_S2['pt_idx']==pt.name,'S2_idx'],:]

            if len(S2_slice)==0 or len(S1_slice)==0:
                return []

            S2_slice['matches'] = S2_slice.apply(lambda el: S1_slice.loc[(S1_slice['beginposition']>(el['beginposition']-timedelta(days=self.CONFIG['day_offset'])))&(S1_slice['beginposition']<(el['beginposition']+timedelta(days=self.CONFIG['day_offset']))),:].index.values, axis=1)
            #print ('pre',S2_slice.columns, S2_slice.index.name)
            S2_slice = S2_slice.explode('matches').reset_index()
            S2_slice = S2_slice.loc[~S2_slice['matches'].isna()]
            #print ('post',S2_slice.columns, S2_slice.index.name)
            return S2_slice[['level1cpdiidentifier','matches']].values.tolist()

        S1_tree = pygeos.STRtree([pygeos.io.from_wkt(subshp) for subshp in S1_df['footprint'].values.tolist()])
        S2_tree = pygeos.STRtree([pygeos.io.from_shapely(subshp) for subshp in S2_df['intersection_geom'].values.tolist()])

        Q_S1 = pd.DataFrame(S1_tree.query_bulk(pts['bbox_wgs'].values.tolist(), predicate='within').T, columns=['pt_idx','S1_idx'])
        Q_S2 = pd.DataFrame(S2_tree.query_bulk(pts['bbox_wgs'].values.tolist(), predicate='within').T, columns=['pt_idx','S2_idx'])

        S2_df['beginposition'] = pd.to_datetime(S2_df['beginposition'])
        S1_df['beginposition'] = pd.to_datetime(S1_df['beginposition'])
        S1_df['endposition'] = pd.to_datetime(S1_df['endposition'])
        
        #print ('pts',pts)

        pts['matches'] = pts.apply(apply_matches, axis=1)

        return pts[pts['matches'].str.len()>0]
    
    
if __name__=="__main__":
    #COUNTRY_CODES_EU = ['AT','BE','BG','CY','CZ','DK','EE','FI','FR','DE','GR','HU','IE','IT','LV','LT','LU','MT','NL','PL','PT','RO','SK','SI','ES','SE','GB']
    pass


