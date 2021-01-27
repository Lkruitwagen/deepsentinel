import os, glob, logging
from tqdm import tqdm


class GCPClient:
    """
    A basic class to manage uploads and downloads from GCP storage.

    Attributes
    ----------
    gcp_credentials_path: str
        The path to a json with GCP service account credentials (create here: https://console.cloud.google.com/apis/credentials/serviceaccountkey).
    """
    
    def __init__(self,gcp_credentials_path, storage_bucket, version, make_bucket=False):
        from google.cloud import storage
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_credentials_path
        # Instantiates a client
        self.client = storage.Client()
        
        self.bucket = self.maybe_get_bucket(storage_bucket, make_bucket)
        
        self.version = version
        
        
    def maybe_get_bucket(self, storage_bucket, make_bucket):
        
        if self.client.bucket(storage_bucket).exists() and make_bucket:
            
            raise Exception('Bucket already exists!')
            
        elif not self.client.bucket(storage_bucket).exists() and not make_bucket:
            
            raise Exception('Bucket does not exist!')
            
        elif not self.client.bucket(storage_bucket).exists() and make_bucket:
            
            bucket = self.client.create_bucket(storage_bucket)
            
        else:
            
            bucket = self.client.get_bucket(storage_bucket)
            
        return bucket
            
        
    def upload(self, f):
        
        idx = os.path.split(f)[1].split('_')[0]
        
        blob = self.bucket.blob(os.path.join(self.version, idx, os.path.split(f)[1]))
        
        blob.upload_from_filename(f)
        
        
    def sync_download(self, source_dir, dest_dir):
        
        blobs = [bb for bb in self.client.list_blobs(self.bucket,prefix=source_dir)]
        
        for blob in tqdm(blobs, desc=f'Syncing {source_dir.split("/")[0]}'):
                
            if os.path.exists(os.path.join(dest_dir, blob.name)):
                # if the path exists
                if (os.stat(os.path.join(dest_dir, blob.name)).st_size!=blob.size):
                    # if the sizes aren't the same:
                    fpath = os.path.join(dest_dir, os.path.split(blob.name)[0])
                    if not os.path.exists(fpath):
                        os.makedirs(fpath)
                    blob.download_to_filename(os.path.join(dest_dir, blob.name))
            else: # download
                fpath = os.path.join(dest_dir, os.path.split(blob.name)[0])
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
                blob.download_to_filename(os.path.join(dest_dir, blob.name))
                
                
    def check(self, f):
        
        idx = os.path.split(f)[1].split('_')[0]
        
        return self.bucket.blob(os.path.join(self.version, idx, os.path.split(f)[1])).exists()
        
        
        
class AzureClient:
    """
    A basic class to manage uploads and downloads from MS Azure.

    Attributes
    ----------
    azure_cs_path: str
        The path to a text file with your azure connectionstring (https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#copy-your-credentials-from-the-azure-portal).
    """
    
    def __init__(self,azure_cs_path, version, make_container=False):
        from azure.storage.blob import BlobServiceClient
        
        with open(azure_cs_path, 'r') as f:
            #os.environ['AZURE_STORAGE_CONNECTION_STRING'] = f.read()
            connection_str = f.read()
            
        logger = logging.getLogger("null")
        logger.disabled = True
        # Instantiates a client
        self.client = BlobServiceClient.from_connection_string(connection_str, logger=logger)
        
        self.maybe_make_container(version.lower().replace('_','-'), make_container)
        
        self.version = version.lower().replace('_','-')
        
        
    def maybe_make_container(self, version, make_container=True):
        
        existing_containers = [cc['name'] for cc in self.client.list_containers()]
        
        if version in existing_containers:
            return True
            
        elif version not in existing_containers and not make_container:
            raise Exception('Container does not exist!')
            
        elif version not in existing_containers and make_container:
            self.client.create_container(version)
            
            return True
            
        
    def upload(self, f):
        
        idx = os.path.split(f)[1].split('_')[0]
        
        blob = self.client.get_blob_client(container=self.version, blob=os.path.join(idx, os.path.split(f)[1]))
        
        with open(f, "rb") as data:
            blob.upload_blob(data, overwrite=True)
            
    def check(self,f):
        idx = os.path.split(f)[1].split('_')[0]
        
        blob = self.client.get_blob_client(container=self.version, blob=os.path.join(idx, os.path.split(f)[1]))