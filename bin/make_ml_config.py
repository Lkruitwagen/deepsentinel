import os, yaml

CONFIG = dict(
    
    model_spec='VAE',
    pretrain =True, 
    sacred=dict(
        gcp_bucket='deepsentinel-models',
        gcp_basedir='/'
    ),
    model_config=dict(
        VAE=dict(
            image_channels=5, 
            h_dim=int(((128/2/2/2/2-2)**2)*256), 
            z_dim=32
        ),
    ), 
    pretrain_config=dict(
        VAE=dict(
            LR=0.00005,
            EPOCHS=10,
            BATCH_SIZE=64,
            DATALOADER_WORKERS=6,
            LOG_INTERVAL=4,
        ),
    ), 
    pretrain_loader_config=dict(
        VAE=dict(
            data_config=os.path.join(os.getcwd(),'DATA_CONFIG.yaml'), 
            data_dir='/data/DEMO_unlabelled/', 
            bands=['B2','B3','B4','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
    ),
    encoder_layers=dict(
        VAE=dict(
            meow='cat'
        ),
    ), 
    device='cuda',
    verbose=True
    
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'ML_CONFIG.yaml'),'w'))

