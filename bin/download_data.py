""" A command-line script to download a dataset from a google bucket"""

import argparse, os

from deepsentinel.utils.storageutils import GCPClient

def sync_local(gcp_credentials_path, storage_bucket, source_dir, dest_dir, version=0):
    
    if not os.path.exists(dest_dir):
        os.path.mkdirs(dest_dir)
        
    client = GCPClient(
                gcp_credentials_path, 
                storage_bucket, 
                version, 
                make_bucket=False)
    
    client.sync_download(source_dir, dest_dir)


parser = argparse.ArgumentParser(description='A script to sync a local directory with cloud-stored data.')
parser.add_argument('gcp_credentials', type=str, help='path to local GCP credentials.')
parser.add_argument('bucket', type=str, help='The gcp bucket containing the dataset.')
parser.add_argument('dataset', type=str, help='The dataset to download.')
parser.add_argument('data_root', type=str, help='The destination for the dataset.')
if __name__=="__main__":
    args = parser.parse_args()
    
    sync_local(args.gcp_credentials, args.bucket, args.dataset, args.data_root)