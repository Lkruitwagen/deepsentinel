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
        
        self.maybe_make_container(version.replace('_','-'), make_container)
        
        self.version = version.replace('_','-')
        
        
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