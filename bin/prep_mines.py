import os
import pandas as pd
import geopandas as gpd
from shapely import ops, geometry
from geopy.distance import geodesic 


def map_dist(row):
    pt1,pt2 = ops.nearest_points(row['mine_polygon'],row['nrgi_pt'])
    return geodesic((pt1.y,pt1.x),(pt2.y,pt2.x)).km

def combine_mine_data(conf, buffer):
    
    if conf:
        CONFIG = yaml.load(open(conf,'r'), Loader=yaml.SafeLoader)
    else:    
        CONFIG = yaml.load(open(os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

    # load data
    gdf = gpd.read_file(os.path.join(CONFIG['NE_ROOT'],'NRGI_assets.geojson'))
    mines = gpd.read_file(os.path.join(CONFIG['NE_ROOT'],'global_mining_polygons_v1.gpkg'))

    # buffer in wgs84 degrees
    gdf['buffer_geom'] = gdf.buffer(buffer)
    gdf = gdf.set_geometry('buffer_geom')
    
    # sjoin and tidy
    mines = gpd.sjoin(mines.reset_index(), gdf[['buffer_geom','geometry', 'GOLD', 'COPPER',
       'SZLN', 'COAL', 'IRON', 'URANIUM', 'AGGREGATES', 'OTHER']], op='intersects',how='left')
    mines = mines.rename(columns={'geometry_left':'mine_polygon','geometry_right':'nrgi_pt'})
    
    # filter the matched mines
    mines.loc[~mines['nrgi_pt'].isna(),'min_dist'] = mines.loc[~mines['nrgi_pt'].isna()].apply(lambda row: map_dist(row), axis=1)
    matched_mines = mines.loc[~mines['min_dist'].isna(),:].sort_values('min_dist').groupby('index').nth(0)
    matched_mines = matched_mines.reset_index().sort_values('min_dist').groupby('index_right').nth(0)
    matched_mines = matched_mines.loc[matched_mines['min_dist']<10,:]
    matched_mines = matched_mines.reset_index().rename(columns={'index_right':'index_nrgi','index':'index_maus','mine_polygon':'geometry'}).drop(columns=['nrgi_pt','min_dist'])
    matched_mines['index_maus'] = matched_mines['index_maus'].astype(int)
    matched_mines = matched_mines.set_geometry('geometry')
    
    # reload mines
    mines = gpd.read_file(os.path.join(CONFIG['NE_ROOT'],'global_mining_polygons_v1.gpkg'))
    mines = mines.loc[~mines.index.isin(matched_mines.index)]
    
    # write out to file
    mines.to_file(os.path.join(CONFIG['NE_ROOT'],'mines_maus.gpkg'),driver='GPKG')
    matched_mines.to_file(os.path.join(CONFIG['NE_ROOT'],'mines_nrgi.gpkg'),driver='GPKG')
    
if __name__==__main__:
    combine_mine_data(os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'),0.5)