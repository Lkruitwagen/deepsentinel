import multiprocessing as mp

import pandas as pd

def read_parquet(f):
    return pd.read_parquet(f)

def async_load_parquets(parquet_files, N_workers):
    
    list2tup = [(f,) for f in parquet_files]
    
    with mp.Pool(N_workers) as P:
        dfs = P.starmap(read_parquet, list2tup)
        
    return pd.concat(dfs)