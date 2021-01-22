import os, yaml

CONFIG = dict(
    DATA_ROOT = os.path.join(os.getcwd(),'data'),       # root directory for data
    NE_ROOT = os.path.join(os.getcwd(),'data'),       # root directory for data
    CATALOG_ROOT = os.path.join(os.getcwd(),'data','catalog'), # root directory for catalog data
    POINTS_ROOT = os.path.join(os.getcwd(),'data','pts'),      # root directory for points
    ee_credentials = os.path.join(os.getcwd(),'ee-deepsentinel.json'),
    gcp_credentials_path = os.path.join(os.getcwd(),'deepsentinel-gcp-key.json'), # json key 
    gcp_storage_bucket = 'deepsentinel',             # storage bucket
    azure_path = os.path.join(os.getcwd(),'azure-connectionstring.txt'), # connection string
    scihub_auth=os.path.join(os.getcwd(),'credentials.json'), 
    osm_credentials = os.path.join(os.getcwd(),'osm_credentials.json'),
    max_date = '2020-07-31',                            # maximum query data
    min_date = '2019-08-01',                            # minimum query date
    dt_lookup = 30,                                     # datetime lookup window, days
    day_offset = 3,                                     # S1-S2 offset lookup window, days
    selfmatch_offset = 6,                               # window for S2 selfmatch, hours
    resolution = 10,                                    # resolution, m
    patch_size = 256,                                    # patch size, pix                                            
    mosaic = False,                                     # mosaic the matching images
    orbit_period=12,                                    # orbit period for S1
    N_workers = 4,                                      # N_workers for async downloads

    DL = dict(
    	S1_bands = ['vv','vh'],
    	S2_bands = ['coastal-aerosol','blue','green','red','red-edge','red-edge-2','red-edge-3','nir','red-edge-4','water-vapor','cirrus','swir1','swir2','alpha'],
    ),
	GEE = dict(
		S2_bands = ['B1','B2', 'B3', 'B4','B5','B6','B7','B8','B8A','B9','B11','B12'],
		S1_bands = ['VV','VH'],
	),
    DL_LC = dict(
        LC_product = 'oxford-university:corine-land-cover',
        LC_bands = ['CLC_class'],
        LC_start_date = '2018-12-01',
        LC_end_date = '2019-01-05',
        legend_json = os.path.join(os.getcwd(),'data','CLC_100m_colmap.json'),
    ),
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'),'w'))