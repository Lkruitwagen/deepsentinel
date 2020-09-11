"""
A class to generate a set of points for the imagery downloader.
"""
import os, yaml, time, random, json
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
import geopandas as gpd
import numpy as np
import pygeos
from shapely import wkt, geometry
from area import area

gpd.options.use_pygeos = True

from deepsentinel.utils.download_catalog import sentinel_products_mpcaller, sentinel_products_worker
from deepsentinel.utils.geoutils import pt2bbox_wgs

class PointGenerator:

    def __init__(self):
        self.CONFIG = yaml.load(open(os.path.join(os.getcwd(),'CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

        countries = gpd.read_file(os.path.join(self.CONFIG['DATA_ROOT'],'ne_10m_countries.gpkg'))
        self.countries = countries[~countries['geometry'].isna()]
        
        self.tree = pygeos.STRtree([pygeos.io.from_shapely(subshp) for subshp in list(countries['geometry'].values)])
        
        self.S2_tiles = gpd.read_file(os.path.join(self.CONFIG['DATA_ROOT'],'S2_utm.gpkg'))

    # approach
    # 1. download the catalog
    # 2. for each timestep, generate the number of points required
    #   - sample from land
    #   - get the trees
    #   - get the spatial/temporal intersection
    #   - get the DL product name, get the GEE product name
    #   - save to disk



    def main_generator(self,start_date, N_orbits, N_points):


        ### Download catalog
        period=12 # days
        N_periods = 30 # 360 days
        OFFSET = 3 # days within to match S1 S2

        auth = json.load(open(os.path.join(os.getcwd(),'credentials.json'),'r'))

        N_workers = 3
        

        ### figure out how to do this better - maybe download, daily slices? download on the fly? etc.
        #sentinel_products_mpcaller(start_date, self.CONFIG['orbit_period'], N_orbits, auth, N_workers)

        for orbit in range(N_orbits):
            
            min_date = start_date + timedelta(days=period*orbit)
            max_date = start_date + timedelta(days=period*(orbit+1))

            S1_df = pd.read_parquet(os.path.join(os.getcwd(),'data','catalog',f'S1_{str(orbit)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))
            S2_L2A_df = pd.read_parquet(os.path.join(os.getcwd(),'data','catalog',f'S2_L2A_{str(orbit)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))
            S2_L1C_df = pd.read_parquet(os.path.join(os.getcwd(),'data','catalog',f'S2_L1C_{str(orbit)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))

            S2_L1C_df['beginposition'] = pd.to_datetime(S2_L1C_df['beginposition'])
            S2_L2A_df['beginposition'] = pd.to_datetime(S2_L2A_df['beginposition'])
            S1_df['beginposition'] = pd.to_datetime(S1_df['beginposition'])
            S1_df['endposition'] = pd.to_datetime(S1_df['endposition'])
            
            # only retain S2 records where there is both L1C and L2A
            S2_L2A_df = S2_L2A_df[S2_L2A_df['level1cpdiidentifier'].isin(S2_L1C_df['level1cpdiidentifier'])]
            S2_L2A_df = S2_L2A_df.set_index('level1cpdiidentifier')
            S2_L1C_df = S2_L1C_df.set_index('level1cpdiidentifier')
            


            pts = pd.DataFrame(columns=['lon','lat','bbox_wgs','matches'])

            print ('getting points',end='..')
            while len(pts)<N_points:

                # obtain points
                new_pts = self._sample_land_points(N_points-len(pts))

                new_pts['bbox_wgs'] = new_pts.apply(lambda pt: pt2bbox_wgs(pt, use_pygeos=True), axis=1)

                # add matches
                new_pts = self._get_matches(new_pts, S1_df, S2_L2A_df, OFFSET)

                pts = pts.append(new_pts)
                print (len(pts),end='..')

            # match pts to DL and GEE
            pts['matches'] = pts['matches'].apply(random.choice)
            pts['S1_rec'] = pts['matches'].apply(lambda match: S1_df.loc[match[1],:].to_dict())
            pts['S2_L2A_rec'] = pts['matches'].apply(lambda match: S2_L2A_df.loc[match[0],:].to_dict())
            pts['S2_L1C_rec'] = pts['matches'].apply(lambda match: S2_L1C_df.loc[match[0],:].to_dict())

            pts['DL_S1'] = pts.apply(self._map_DL_S1, axis=1)
            pts['DL_S2'] = pts.apply(self._map_DL_S2, axis=1)
            pts['GEE_S1'] =  pts['S1_rec'].apply(lambda el: 'projects/earthengine-public/assets/COPERNICUS/S1_GRD/'+el['title'])
            pts['GEE_S2'] = pts.apply(self._map_GEE_S2, axis=1)
            
            print (pts)
    
    def _map_DL_S1(self,el):
        base='sentinel-1:GRD:meta'
        mean_dt = el['S1_rec']['beginposition']+(el['S1_rec']['endposition']-el['S1_rec']['beginposition'])/2
        dd=mean_dt.isoformat()[0:10]
        meta=f'{el["S1_rec"]["relativeorbitnumber"]:03d}{el["S1_rec"]["orbitdirection"][0]}{mean_dt.hour}{mean_dt.minute}'
        satellite = el['S1_rec']['title'].split('_')[0]
        
        return '_'.join([base,dd,meta,satellite])

    def _map_DL_S2(self,el):
        base='sentinel-2:L1C:'
        mean_dt = el['S2_L1C_rec']['beginposition']+(el['S2_L1C_rec']['endposition']-el['S2_L1C_rec']['beginposition'])/2
        dd=mean_dt.isoformat()[0:10]
        utm_tile = el['S2_L1C_rec']['title'].split('_')[5][1:]
        satellite = el['S2_L1C_rec']['title'].split('_')[0]
        
        # do coverage
        geom1 = self.S2_tiles[self.S2_tiles.Name==utm_tile].iloc[0]['geometry']
        geom2 = wkt.loads(el['S2_L1C_rec']['footprint'])
        
        coverage = area(geometry.mapping(geom2.intersection(geom1)))/area(geometry.mapping(geom1))
        
        return base+'_'.join([dd,utm_tile,f'{int(coverage*100)}',satellite,'v1'])
    
    def _map_GEE_S2(self,el):
        base = 'projects/earthengine-public/assets/COPERNICUS/S2_SR/'
        dt0 = el['S2_L2A_rec']['s2datatakeid'].split('_')[1]
        dt1 = el['S2_L1C_rec']['datastripidentifier'].split('_')[-2][1:]
        utm_tile = el['S2_L2A_rec']['title'].split('_')[5][1:]
        return base+'_'.join([dt0,dt1,utm_tile])
        
            

    def _sample_land_points(self,N_pts):

        pt_df = pd.DataFrame(columns=['lon','lat'], index=[])

        print ('sampling pts',end='..')
        
        ii_p=0
        while len(pt_df)<N_pts:
            print (ii_p,end='..')# can into multip?
            tic = time.time()

            pts = np.random.rand(2*2*(N_pts-len(pt_df))).reshape((N_pts-len(pt_df))*2,2)
            #print ('pts shape',pts.shape)
            pts[:,0] = pts[:,0] * 360 - 180
            pts[:,1] = pts[:,1] * 180 - 90

            pts_pygeos = pygeos.points(pts)

            Q = self.tree.query_bulk(pts_pygeos, predicate='within').T[:,0]
            #print ('Q-shape',Q.shape)


            pt_df = pt_df.append(pd.DataFrame(pts[Q], columns=['lon','lat']))
            #print ('len pt_df', len(pt_df), 'iter_toc',time.time()-tic)
            ii_p+=1

        pt_df = pt_df.iloc[0:N_pts]

        return pt_df


    def _get_matches(self, pts, S1_df, S2_df, OFFSET):

        def apply_matches(pt):
            S1_slice = S1_df.iloc[Q_S1.loc[Q_S1['pt_idx']==pt.name,'S1_idx'],:]
            S2_slice = S2_df.iloc[Q_S2.loc[Q_S2['pt_idx']==pt.name,'S2_idx'],:]

            if len(S2_slice)==0 or len(S1_slice)==0:
                return []

            S2_slice['matches'] = S2_slice.apply(lambda el: S1_slice.loc[(S1_slice['beginposition']>(el['beginposition']-timedelta(days=OFFSET)))&(S1_slice['beginposition']<(el['beginposition']+timedelta(days=OFFSET))),:].index.values, axis=1)
            S2_slice = S2_slice.explode('matches').reset_index()
            S2_slice = S2_slice.loc[~S2_slice['matches'].isna()]
            return S2_slice[['level1cpdiidentifier','matches']].values.tolist()

        S1_tree = pygeos.STRtree([pygeos.io.from_wkt(subshp) for subshp in S1_df['footprint'].values.tolist()])
        S2_tree = pygeos.STRtree([pygeos.io.from_wkt(subshp) for subshp in S2_df['footprint'].values.tolist()])

        Q_S1 = pd.DataFrame(S1_tree.query_bulk(pts['bbox_wgs'].values.tolist(), predicate='within').T, columns=['pt_idx','S1_idx'])
        Q_S2 = pd.DataFrame(S2_tree.query_bulk(pts['bbox_wgs'].values.tolist(), predicate='within').T, columns=['pt_idx','S2_idx'])

        S2_df['beginposition'] = pd.to_datetime(S2_df['beginposition'])
        S1_df['beginposition'] = pd.to_datetime(S1_df['beginposition'])
        S1_df['endposition'] = pd.to_datetime(S1_df['endposition'])

        pts['matches'] = pts.apply(apply_matches, axis=1)

        return pts[pts['matches'].str.len()>0]
    
    
if __name__=="__main__":
    generator=PointGenerator()
    generator.main_generator(dt(2019,8,1,0,0), 2, 30)


