import pyproj
from math import ceil
from functools import partial
from itertools import product
from shapely import geometry, ops, wkt
import pygeos

def pt2bbox_wgs(pt, patch_size, resolution, use_pygeos=False):
    #print (pt)
    utm_zone = get_utm_zone(pt['lat'], pt['lon'])
    proj_utm = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')
    proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    reproj_wgs_utm = partial(pyproj.transform, proj_wgs, proj_utm)
    reproj_utm_wgs = partial(pyproj.transform, proj_utm, proj_wgs)

    pt_utm = ops.transform(reproj_wgs_utm, geometry.Point(pt['lon'],pt['lat']))
    
    bbox_utm = geometry.box(pt_utm.x-(patch_size*resolution)/2, pt_utm.y-(patch_size*resolution)/2, pt_utm.x+(patch_size*resolution)/2, pt_utm.y+(patch_size*resolution)/2)
    
    bbox_wgs = ops.transform(reproj_utm_wgs, bbox_utm)

    if use_pygeos:
        return pygeos.io.from_shapely(bbox_wgs)
    else:
        return bbox_wgs


def get_utm_zone(lat,lon):
    """A function to grab the UTM zone number for any lat/lon location
    """
    zone_str = str(int((lon + 180)/6) + 1)

    if ((lat>=56.) & (lat<64.) & (lon >=3.) & (lon <12.)):
        zone_str = '32'
    elif ((lat >= 72.) & (lat <84.)):
        if ((lon >=0.) & (lon<9.)):
            zone_str = '31'
        elif ((lon >=9.) & (lon<21.)):
            zone_str = '33'
        elif ((lon >=21.) & (lon<33.)):
            zone_str = '35'
        elif ((lon >=33.) & (lon<42.)):
            zone_str = '37'

    return zone_str

def wgsgeom2utmtiles(geom, patch_size, resolution, use_pygeos=False):
    
    # get the utm zone and geom_utm
    utm_zone = get_utm_zone(geom.centroid.y, geom.centroid.x)
    proj_utm = pyproj.Proj(proj='utm',zone=utm_zone,ellps='WGS84')
    proj_wgs = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    reproj_wgs_utm = partial(pyproj.transform, proj_wgs, proj_utm)
    reproj_utm_wgs = partial(pyproj.transform, proj_utm, proj_wgs)
    geom_utm = ops.transform(reproj_wgs_utm, geom)
    
    # get the covering utm tiles
    D_x = geom_utm.bounds[2]-geom_utm.bounds[0] #m
    D_y = geom_utm.bounds[3]-geom_utm.bounds[1] #m
    d = patch_size*resolution
    N_x = ceil(D_x / d)
    N_y = ceil(D_y / d)
    
    utm_tiles = [geometry.box(
                    geom_utm.bounds[0]+d*ix,
                    geom_utm.bounds[1]+d*iy,
                    geom_utm.bounds[0]+d*(ix+1),
                    geom_utm.bounds[1]+d*(iy+1)
                )
                for ix, iy in product(range(N_x), range(N_y))]
    
    # filter utm tiles if they don't intersect the geometry
    utm_tiles = [t for t in utm_tiles if t.intersects(geom_utm)]
    
    return proj_utm, reproj_utm_wgs, utm_tiles
    
    