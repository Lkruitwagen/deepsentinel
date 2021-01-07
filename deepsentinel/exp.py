"""A script to instantiate the experiment object"""
import os, glob, yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import GoogleCloudStorageObserver

# generate NAME
### todo: get from globbing experiments dir
EX_NAME='test'

CONFIG = yaml.load(open(os.path.join(os.getcwd(),'ML_CONFIG.yaml'),'r'), Loader=yaml.SafeLoader)

ex = Experiment(EX_NAME)
ex.add_config(os.path.join(os.getcwd(),'ML_CONFIG.yaml'))
ex.observers.append(FileStorageObserver('experiments/sacred'))
ex.observers.append(GoogleCloudStorageObserver(bucket=CONFIG['sacred']['gcp_bucket'], basedir=CONFIG['sacred']['gcp_basedir']))
