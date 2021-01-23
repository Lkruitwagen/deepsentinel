"""A script to instantiate the experiment object"""
import os, glob, yaml
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.observers import GoogleCloudStorageObserver

# todo: generate NAME
EX_NAME='test'

ex = Experiment(EX_NAME)
