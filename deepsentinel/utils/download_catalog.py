import json, os
import pandas as pd
from sentinelsat.sentinel import SentinelAPI
from datetime import date, timedelta

import multiprocessing as mp

def async_product_caller(arg_tuples, N_workers):
    
    with mp.Pool(N_workers) as p:
        p.starmap(async_product_worker, arg_tuples)

def async_product_worker(platform, product, date, auth):
    api = SentinelAPI(auth['U'], auth['P'])
    
    products = api.query(date=(date, date+timedelta(days=1)),
                            platformname=platform,
                            producttype=product)

    df = pd.DataFrame.from_dict(products, orient='index')
    
    df.to_parquet(os.path.join(os.getcwd(),'data','catalog','_'.join([platform, product, date.isoformat()[0:10]])+'.parquet'))

def period_mpcaller(start_date, period, N_periods, auth, N_workers):

    args = [(start_date, period, ii_t, auth['scihub']) for ii_t in range(N_periods)]

    with mp.Pool(N_workers) as p:
        p.starmap(async_product_worker, args)
    
def period_worker(start_date, period, ii_t, auth):
    api = SentinelAPI(auth['U'], auth['P'])
    
    min_date = start_date + timedelta(days=period*ii_t)
    max_date = start_date + timedelta(days=period*(ii_t+1))
    
    S2_L2A_products = api.query(date=(min_date, max_date),
                            platformname='Sentinel-2',
                            producttype='S2MSI2A')

    S2_L2A_df = pd.DataFrame.from_dict(S2_products, orient='index')

    S2_L2A_df.to_parquet(os.path.join(os.getcwd(),'data','catalog',f'S2_L2A_{str(ii_t)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))
    
    S2_L1C_products = api.query(date=(min_date, max_date),
                            platformname='Sentinel-2',
                            producttype='S2MSI1C')

    S2_L1C_df = pd.DataFrame.from_dict(S2_L1C_products, orient='index')

    S2_L1C_df.to_parquet(os.path.join(os.getcwd(),'data','catalog',f'S2_L1C_{str(ii_t)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))
                            
    S1_products = api.query(date=(min_date, max_date),
                            platformname='Sentinel-1',
                            producttype='GRD')            

    S1_df = pd.DataFrame.from_dict(S1_products, orient='index')
    
    S1_df.to_parquet(os.path.join(os.getcwd(),'data','catalog',f'S1_{str(ii_t)}_{min_date.isoformat()[0:10]}_{max_date.isoformat()[0:10]}.parquet'))

if __name__=="__main__":
    start_date = date(2019,8,1)
    period=12 # days
    N_periods = 3 # 360 days

    auth = json.load(open(os.path.join(os.getcwd(),'credentials.json'),'r'))

    N_workers = 3

    period_mpcaller(start_date, period, N_periods, auth, N_workers)
