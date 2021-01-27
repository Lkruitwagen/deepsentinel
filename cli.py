import click, os, logging, yaml, json
from datetime import datetime as dt
from click import command, option, Option, UsageError

logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--conf', default=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), help='path to DATA_CONFIG.yaml')
@click.option('--n-orbits', help='The number of orbits to spread the simulated points over.', type=int) # one of N_orbits, end_date, pts per orbit
@click.option('--end-date', help='the end date to stop sampling points, as YYYY-mm-dd', type=str)
@click.option('--iso2', help='A comma-separated list of iso-a2 country codes for geographic subsampling', type=str)
@click.argument('start_date', type=str)
@click.argument('n-points', type=int)
@click.argument('name', type=str)
def generate_points(name, n_points, start_date, iso2, end_date, n_orbits, conf):
    """
    Seed points for a new dataset.
    
    \b
    PARAMETERS
    ----------
    NAME: str
        The name of the new dataset.
        
    N_POINTS: int
        The number of data points to generate.
        
    START_DATE: str
        The start date for data collection in the form YYYY-mm-dd.
    """
    
    from deepsentinel.utils.point_generator import PointGenerator
    logger = logging.getLogger('GENERATE_POINTS')

    
    # error check either end_date OR n_orbits
    assert (end_date or n_orbits), 'Only one of n_orbits or end_date must be provided.'
    assert not (end_date and n_orbits), 'Only one of n_orbits or end_date must be provided.'
    
    # error check date formats
    try:
        start_date = dt.strptime(start_date,'%Y-%m-%d')
    except:
        raise ValueError('Ensure start_date is in the correct format, YYYY-mm-dd')
    if end_date!=None:
        try:
            end_date = dt.strptime(end_date,'%Y-%m-%d')
        except:
            raise ValueError('Ensure end_date is in the correct format, YYYY-mm-dd')
                   
    logger.info('Generating points with:')
    logger.info(f'NAME:{name}')
    logger.info(f'N_POINTS:{n_points}')
    logger.info(f'START_DATE:{start_date}')
    logger.info(f'iso2:{iso2}')
    logger.info(f'end_date:{end_date}')
    logger.info(f'n_orbits:{n_orbits}')
    logger.info(f'conf:{conf}')
    
    if iso2:
        iso2 = iso2.split(',')
    
    logger.info('Initialising generator')
    generator=PointGenerator(iso_geographies=iso2, conf=conf)
    
    if not n_orbits: # get n_orbits from end_date
        n_orbits = (end_date-start_date).days // generator.CONFIG['orbit_period']
        pts_per_orbit = n_points//n_orbits + 1
    else: # have n_orbits, get pts_per_orbit
        pts_per_orbit = n_points//n_orbits + 1
        
    logger.info(f'Running generator for {name} from {start_date.isoformat()} for {n_orbits} orbits with {pts_per_orbit} points per orbit')
    generator.main_generator(start_date, n_orbits, pts_per_orbit,name)
    
    
@cli.command()
@click.option('--conf', default=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), help='path to DATA_CONFIG.yaml')
@click.argument('gdf_path', type=str)
@click.argument('name', type=str)
@click.argument('start_date', type=str)
@click.argument('end-date', type=str)
def geopandas_to_points(gdf_path, name, start_date, end_date, conf):
    """
    Seed points for a new dataset.
    
    \b
    PARAMETERS
    ----------
    GDF_PATH: str
        The path to the GeoPandas GeoDataFrame to load (with gpd.read_file).
        
    NAME: str
        The name of the new dataset.
        
    START_DATE: str
        The start date for data collection in the form YYYY-mm-dd.
        
    END_DATE: str
        The end date for data collection in the form YYYY-mm-dd.
    """
    
    from deepsentinel.utils.gdf2points import GDF2Points
    logger = logging.getLogger('POINTS_FROM_GDF')

    # error check date formats
    try:
        start_date = dt.strptime(start_date,'%Y-%m-%d')
    except:
        raise ValueError('Ensure start_date is in the correct format, YYYY-mm-dd')
    try:
        end_date = dt.strptime(end_date,'%Y-%m-%d')
    except:
        raise ValueError('Ensure end_date is in the correct format, YYYY-mm-dd')
                   
    logger.info('Generating points with:')
    logger.info(f'GDF_PATH:{gdf_path}')
    logger.info(f'NAME:{name}')
    logger.info(f'START_DATE:{start_date}')
    logger.info(f'END_DATE:{end_date}')
    logger.info(f'conf:{conf}')
    
    logger.info('Initialising generator')
    generator=GDF2Points(conf=conf)
    

        
    logger.info(f'Running generator for {name} sampling tiles for {gdf_path} from {start_date.isoformat()} to {end_date.isoformat()}')
    generator.generate_from_gdf(gdf_path, start_date, end_date, name)

    
@cli.command()
@click.option('--conf', default=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), help='path to DATA_CONFIG.yaml')
@click.argument('name', type=str)
@click.argument('sources', type=str)
@click.argument('destinations', type=str)
def generate_samples(name, sources, destinations, conf):
    """
    Download imagery samples for a seeded dataset.
    
    \b
    PARAMETERS
    ----------
    NAME: str
        The name of the dataset to download.
        
    SOURCES: str
        A comma-separated list of sources to download the matching data from. Must be in ['dl','gee','osm','clc']:
            dl: DescartesLabs (https://www.descarteslabs.com/)
            gee: Google Earth Engine (https://earthengine.google.com/)
            osm: OpenStreetMap (https://www.openstreetmap.org/, https://github.com/Lkruitwagen/deepsentinel-osm)
            clc: Copernicus Land Cover (https://land.copernicus.eu/pan-european/corine-land-cover, mirrored on DescartesLabs)
        
    DESTINATIONS: str
        A comma-separated list of desintations for the generated data. Must be in ['local','gcp','azure']:
            local: saved to <data_root>/<name>/
            gcp: saved to a Google Cloud Storage Bucket
            azure: saved to an Azure Cloud Storage Container
    """
    
    from deepsentinel.utils.sample_generator import SampleDownloader
    logger = logging.getLogger('SAMPLE_IMAGERY')
    
    # error check destinations and sources
    for source in sources.split(','):
        assert (source in ['dl','gee','osm','clc'])
    sources = sources.split(',')
    for dest in destinations.split(','):
        assert (dest in ['local','gcp','azure'])
    destinations = destinations.split(',')
                       
    logger.info('Sampling imagery with:')
    logger.info(f'NAME:{name}')
    logger.info(f'SOURCES:{sources}')
    logger.info(f'DESTINATIONS:{destinations}')
    
    
    
    
    downloader=SampleDownloader(version=name, destinations=destinations, conf=conf)
    if 'dl' in sources:
        logger.info('doing dl')
        downloader.download_samples_DL()
    if 'gee' in sources:
        logger.info('gee')
        downloader.download_samples_GEE()
    if 'lc' in sources:
        downloader.download_samples_LC()
    if 'osm' in sources:
        downloader.download_samples_OSM()
        
    logger.info('DONE!')


    
@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,    
    )
)
@click.option('--conf', default=os.path.join(os.getcwd(),'conf','ML_CONFIG.yaml'), help='path to ML_CONFIG.yaml')
@click.option('--observers', default='local,gcp', help='Comma-separated list of observers to add to experiment, from ["local","gcp"]')
@click.option('--name', )
@click.pass_context
def train(ctx, conf, observers, name):
    """
    Run the model training scripts with Sacred and a YAML config file. 
    
    \b
    Any additional parameters can also be specified:
    --device=cuda
    
    Nested parameters can be specified like so:
    --model_config--VAE--z_dim=16
    --model_config--VAE={\"z_dim\":16}
    """
    from deepsentinel.utils.utils import get_from_dict, set_in_dict, make_nested_dict
    from deepsentinel.main import ex
    from sacred.observers import FileStorageObserver
    from sacred.observers import GoogleCloudStorageObserver
    
    logger = logging.getLogger('TRAINING')
    
    CONFIG = yaml.load(open(conf,'r'), Loader=yaml.SafeLoader)
    
    logger.info(f'Adding config from {conf}')
    ex.add_config(conf)
    
    ctx_conf = {}
    
    for item in ctx.args:
        kks,vv = item.split('=')
        kks = kks.split('--')[1:]
        
        # cast vv to the type from the nested config
        vv_type = type(get_from_dict(CONFIG,kks))   
        #override special cases
        if vv_type==dict:
            vv = json.loads(vv)
        elif kks[0] in ['pretrain', 'finetune','load_run'] and vv=='None':
            vv = None
        elif kks[0]=='load_run':
            vv = int(vv)
        elif kks[0] in ['pretrain','finetune']:
            vv = str(vv)
        else:
            vv = vv_type(vv)
        
        # set in our update dict
        try:
            set_in_dict(ctx_conf,kks,vv)
        except:
            for ii_k in range(1,len(kks)):
                try:
                    set_in_dict(ctx_conf,kks[:-ii_k],make_nested_dict(kks[-1*ii_k:],vv))
                    break
                except:
                    pass
        
    logger.info(f'Adding additional CLI config: {ctx_conf}')
        
    # add observers
    if 'local' in observers:
        logger.info(f'Adding local observer at {CONFIG["sacred"]["local"]}')
        ex.observers.append(FileStorageObserver(CONFIG['sacred']['local']))
    if 'gcp' in observers:
        logger.info(f'Adding Google Cloud Observer at {CONFIG["sacred"]["gcp_bucket"]}/{CONFIG["sacred"]["gcp_basedir"]}')
        ex.observers.append(GoogleCloudStorageObserver(bucket=CONFIG['sacred']['gcp_bucket'], basedir=CONFIG['sacred']['gcp_basedir']))
        
    r = ex.run(config_updates=ctx_conf)



if __name__=="__main__":
    cli()

