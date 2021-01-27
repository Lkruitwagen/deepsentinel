import json, requests, os, pygeos, logging, geojson, pyproj, io
import pandas as pd
import geopandas as gpd
gpd.use_pygeos=True
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from deepsentinel.utils.storageutils import GCPClient, AzureClient


def OSM_downloader(version, pts, ii_ps, CONFIG, mp_idx, TLs, destinations):
    osm_credentials = json.load(open(CONFIG['osm_credentials'],'r'))
    
    logger_mp = logging.getLogger(f'OSM_{mp_idx}')
    
    # spatial join continent
    continents = gpd.read_file(os.path.join(os.getcwd(),'data','continents.gpkg'))
    
    pts['geometry'] = pygeos.creation.points(pts['lon'].values,pts['lat'].values)
    pts = gpd.sjoin(gpd.GeoDataFrame(pts, geometry=pts['geometry']), continents[['continent','geometry']], how='left', op='within').reset_index().groupby('idx').first()

    pts = pd.DataFrame(pts)
    
    if 'gcp' in destinations:
        gcp_client = GCPClient(CONFIG['gcp_credentials_path'],CONFIG['gcp_storage_bucket'],version)
        
    if 'azure' in destinations:
        azure_client = AzureClient(CONFIG['azure_path'], version, make_container=False)
        
    for idx, pt in pts.iterrows():
        
        # make the path
        if not os.path.exists(os.path.join(CONFIG['DATA_ROOT'], version,str(idx))):
            os.makedirs(os.path.join(CONFIG['DATA_ROOT'], version,str(idx)))
            
            
        # get the tile
        tile = TLs[str(idx)]

        
        try:
            
            tile['properties']['continent'] = pt['continent']
            
            fts = []
            
            for geom_type in ['points','lines','polygons']:
                

                body = {'feature':json.dumps(tile),'geom_type':geom_type}
                resp = requests.post(osm_credentials['url'], body, auth=(osm_credentials['U'],osm_credentials['P']))
                results = json.loads(resp.text)
                
                fts = fts + results['features']


            if 'local' in destinations:
                name_root = os.path.join(CONFIG['DATA_ROOT'],version,str(idx),'_'.join([str(idx),'OSMFC',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

            else:
                name_root = os.path.join(CONFIG['DATA_ROOT'],'tmp','_'.join([str(idx),'OSMFC',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))
                
            json.dump(geojson.FeatureCollection(fts), open(name_root+'.geojson','w'))

            
            if 'gcp' in destinations:
                gcp_client.upload(name_root+'.geojson')
                
            if 'azure' in destinations:

                azure_client.upload(name_root+'.geojson')
                
            if 'local' not in destinations:

                os.remove(name_root+'.geojson')
                    
            
                

            print (f'done pt {idx}, N_fts: {len(fts)}, latlon: {pt["lon"]},{pt["lat"]}')
        except Exception as e:
            print ('Error!')
            print (e)

    
    return ii_ps
        
        

def DL_CLC_downloader(version, pts, ii_ps, CONFIG, mp_idx, TLs, destinations):
    
    import descarteslabs as dl
    raster_client = dl.Raster()
    
    if 'gcp' in destinations:
        gcp_client = GCPClient(CONFIG['gcp_credentials_path'],CONFIG['gcp_storage_bucket'],version)
        
    if 'azure' in destinations:
        azure_client = AzureClient(CONFIG['azure_path'], version, make_container=False)
    
    logger_mp = logging.getLogger(f'LC_{mp_idx}')
    
    colmap = json.load(open(CONFIG['DL_LC']['legend_json'],'r'))
    
    def _save_LC_thumbnail(arr, path):

        # change arr to colmap
        im_arr = np.zeros((*arr.shape,3))
        for kk in colmap.keys():

            im_arr[arr==int(kk),0] = colmap[kk]['color'][0]
            im_arr[arr==int(kk),1] = colmap[kk]['color'][1]
            im_arr[arr==int(kk),2] = colmap[kk]['color'][2]

        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.imshow(im_arr)
        ax.axis('off')
        fig.savefig(path,bbox_inches='tight',pad_inches=0)
        plt.close()
    
    for idx, pt in pts.iterrows():
        
        # make the path
        if not os.path.exists(os.path.join(CONFIG['DATA_ROOT'], version,str(idx))):
            os.makedirs(os.path.join(CONFIG['DATA_ROOT'], version,str(idx)))
            
            
        # get the tile
        tile = TLs[str(idx)]

        
        try:

            LC_scenes, LC_ctx = dl.scenes.search(
                                    aoi=tile['geometry'],
                                    products=CONFIG['DL_LC']['LC_product'],
                                    start_datetime=CONFIG['DL_LC']['LC_start_date'],
                                    end_datetime=CONFIG['DL_LC']['LC_end_date']
                                )

            # mosaic
            for ii_s, s in enumerate(LC_scenes):
                print (s.properties['id'])

                s_arr, LC_meta = raster_client.ndarray(s.properties['id'], 
                                                bands=CONFIG['DL_LC']['LC_bands'], 
                                                scales = [(0,255,0,255)]*len(CONFIG['DL_LC']['LC_bands']),
                                                data_type='Byte',
                                                dltile=tile['properties']['key'], 
                                                )

                if ii_s==0:
                    LC_arr = s_arr
                    LC_arr[LC_arr>=48]=0

                else:
                    LC_arr[LC_arr==0] = s_arr[LC_arr==0] 
                    LC_arr[LC_arr>=48]=0


            if 'local' in destinations:
                name_root = os.path.join(CONFIG['DATA_ROOT'],version,str(idx),'_'.join([str(idx),'LC',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

            else:
                name_root = os.path.join(CONFIG['DATA_ROOT'],'tmp','_'.join([str(idx),'LC',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))
                
            np.savez(name_root+'_LCarr.npz', arr=LC_arr.astype(np.uint8))
            json.dump(LC_meta, open(name_root+'_LCmeta.json','w'))

            _save_LC_thumbnail(LC_arr, name_root+'_LCthumb.png')
            
            if 'gcp' in destinations:
                for ext in ['_LCarr.npz','_LCmeta.json','_LCthumb.png']:
                    gcp_client.upload(name_root+ext)
                
            if 'azure' in destinations:
                for ext in ['_LCarr.npz','_LCmeta.json','_LCthumb.png']:
                    azure_client.upload(name_root+ext)
                
            if 'local' not in destinations:
                for ext in ['_LCarr.npz','_LCmeta.json','_LCthumb.png']:
                    os.remove(name_root+ext)
                    
            
                

            print (f'done pt {idx}, arr_min: {LC_arr.min()}, arr_max: {LC_arr.max()}, latlon: {pt["lon"]},{pt["lat"]}')
        except Exception as e:
            print ('Error!')
            print (e)

    
    return ii_ps
    

def DL_downloader(version, pts, ii_ps, CONFIG, mp_idx, destinations):
    
    import descarteslabs as dl
    raster_client = dl.Raster()
    
    if 'gcp' in destinations:
        gcp_client = GCPClient(CONFIG['gcp_credentials_path'],CONFIG['gcp_storage_bucket'],version)
        
    if 'azure' in destinations:
        azure_client = AzureClient(CONFIG['azure_path'], version, make_container=False)
    
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
            
        print (f'checking pt {idx}...', end='')
            
        # get the name
        if 'local' in destinations:
            name_root = os.path.join(CONFIG['DATA_ROOT'],version,str(idx),'_'.join([str(idx),'DL',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

        else:
            name_root = os.path.join(CONFIG['DATA_ROOT'],'tmp','_'.join([str(idx),'DL',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))
            
            
        # check the work done and skip if necessary
        checkers = []
        if 'gcp' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(gcp_client.check(name_root+ext))

        if 'azure' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(azure_client.check(name_root+ext))

        if 'local' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(os.path.exists(name_root+ext))
            
        if not np.product(checkers):
            # if all checkers true, do everything            
            
            # get the tile
            tile = raster_client.dltile_from_latlon(pt['lat'],pt['lon'], CONFIG['resolution'], CONFIG['patch_size'],0)

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
                    S2_max_date = S2_min_date + timedelta(days=3)

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
                    S1_max_date = S1_min_date + timedelta(days=2)

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


                np.savez(name_root+'_S2arr.npz', arr=S2_arr.astype(np.float32))
                json.dump(S2_meta, open(name_root+'_S2meta.json','w'))
                np.savez(name_root+'_S1arr.npz', arr=S1_arr.astype(np.float32))
                json.dump(S1_meta, open(name_root+'_S1meta.json','w'))
                json.dump(tile, open(name_root+'_tile.json','w'))

                _save_thumbnail((S2_arr[:,:,(3,2,1)].astype(np.float32)/10000).clip(0,1), name_root+'_S2thumb.png')
                _save_thumbnail((np.stack([S1_arr[:,:,0],np.zeros(S1_arr.shape[0:2]),S1_arr[:,:,1]])/255*2.5).clip(0,1).transpose([1,2,0]), name_root+'_S1thumb.png')

                if 'gcp' in destinations:
                    for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                        gcp_client.upload(name_root+ext)

                if 'azure' in destinations:
                    for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                        azure_client.upload(name_root+ext)

                if 'local' not in destinations:
                    for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                        os.remove(name_root+ext)




                print (f'done pt {idx}, S2_min: {S2_arr.min()}, S2_max: {S2_arr.max()}, S1_min: {S1_arr.min()}, S1_max: {S1_arr.max()}, {pt["DL_S2"]},{pt["lon"]},{pt["lat"]}')
                with open(os.path.join(os.getcwd(),'logs',f'{version}_dl.log'), "a") as f:
                    f.write(f"{idx}\n")
                
            except Exception as e:
                print ('Error!')
                print (e)

            TLs[idx] = dict(tile)
        else:
            print (f'got pt {idx} already')
            with open(os.path.join(os.getcwd(),'logs',f'{version}_dl.log'), "a") as f:
                f.write(f"{idx}\n")
                          
    
    return TLs, ii_ps
                
    
    

def GEE_downloader(version, pts, ii_ps, CONFIG, mp_idx, TLs, destinations):

    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    KEY = CONFIG['ee_credentials']

    credentials = service_account.Credentials.from_service_account_file(KEY)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

    session = AuthorizedSession(scoped_credentials)
        
    logger_mp = logging.getLogger(f'GEE_{mp_idx}')
    
    PROJ_WGS = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    
    if 'gcp' in destinations:
        gcp_client = GCPClient(CONFIG['gcp_credentials_path'],CONFIG['gcp_storage_bucket'],version)
        
    if 'azure' in destinations:
        azure_client = AzureClient(CONFIG['azure_path'], version, make_container=False)
    
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
                      
                      
        if 'local' in destinations:
            name_root = os.path.join(CONFIG['DATA_ROOT'],version,str(idx),'_'.join([str(idx),'GEE',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))
        else:
            name_root = os.path.join(CONFIG['DATA_ROOT'],'tmp','_'.join([str(idx),'GEE',pt['DL_S2'].split(':')[2][0:10], str(pt['lon']), str(pt['lat'])]))

                      
        # check the work done and skip if necessary
        checkers = []
        if 'gcp' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(gcp_client.check(name_root+ext))

        if 'azure' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(azure_client.check(name_root+ext))

        if 'local' in destinations:
            for ext in ['_S2arr.npz','_S2meta.json','_S1arr.npz','_S1meta.json','_tile.json','_S2thumb.png','_S1thumb.png']:
                checkers.append(os.path.exists(name_root+ext))
            
        if not np.product(checkers):
                      
        
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


                np.savez(name_root+'_S2arr.npz', arr=S2_arr.astype(np.float32))
                np.savez(name_root+'_S1arr.npz', arr=S1_arr.astype(np.float32))
                json.dump(GEE_meta, open(name_root+'_meta.json','w'))

                _save_thumbnail((S2_arr[:,:,(3,2,1)]/10000).clip(0,1), name_root+'_S2thumb.png')
                _save_thumbnail((((np.stack([S1_arr[:,:,0],np.zeros(S1_arr.shape[0:2]),S1_arr[:,:,1]])).transpose([1,2,0])+np.abs(S1_arr.min()))/(S1_arr.max()-S1_arr.min())).clip(0,1), name_root+'_S1thumb.png')


                if 'gcp' in destinations:
                    for ext in ['_S2arr.npz','_meta.json','_S1arr.npz','_S2thumb.png','_S1thumb.png']:
                        gcp_client.upload(name_root+ext)

                if 'azure' in destinations:
                    for ext in ['_S2arr.npz','_meta.json','_S1arr.npz','_S2thumb.png','_S1thumb.png']:
                        azure_client.upload(name_root+ext)

                if 'local' not in destinations:
                    for ext in ['_S2arr.npz','_meta.json','_S1arr.npz','_S2thumb.png','_S1thumb.png']:
                        os.remove(name_root+ext)

                print (f'Done pt {idx}, S2_min: {S2_arr.min()}, S2_max: {S2_arr.max()}, S1_min: {S1_arr.min()}, S1_max: {S1_arr.max()}')
                with open(os.path.join(os.getcwd(),'logs',f'{version}_gee.log'), "a") as f:
                    f.write(f"{idx}\n")

            except Exception as e:
                print (f'Error pt {idx}: {e}')
            
        else:
            print (f'Already done pt {idx}')
            with open(os.path.join(os.getcwd(),'logs',f'{version}_gee.log'), "a") as f:
                f.write(f"{idx}\n")
        
    return True
