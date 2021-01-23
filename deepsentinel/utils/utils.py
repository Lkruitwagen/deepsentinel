from functools import reduce
import operator
import multiprocessing as mp

import pandas as pd

def read_parquet(f):
    return pd.read_parquet(f)

def async_load_parquets(parquet_files, N_workers):
    
    list2tup = [(f,) for f in parquet_files]
    
    with mp.Pool(N_workers) as P:
        dfs = P.starmap(read_parquet, list2tup)
        
    return pd.concat(dfs)

def get_from_dict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def set_in_dict(dataDict, mapList, value):
    get_from_dict(dataDict, mapList[:-1])[mapList[-1]] = value
    
def make_nested_dict(mapList, value):
    dd = {}
    for ii_k in range(len(mapList)-1):
        set_in_dict(dd,mapList[:(ii_k+1)],{})
    set_in_dict(dd,mapList,value)
    return dd