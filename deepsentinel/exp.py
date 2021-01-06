"""A script to instantiate the experiment object"""
import os, glob
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import GoogleCloudStorageObserver

# generate NAME
### todo: get from globbing experiments dir
NAME='test'

ex = Experiment(NAME)

ex.observers.append(FileStorageObserver('experiments'))
ex.observers.append(GoogleCloudStorageObserver(bucket=gcp_conf['bucket'], basedir=gcp_conf['basedir']))
ex.add_config(os.path.join(os.getcwd(),'ML_CONFIG.yaml'))