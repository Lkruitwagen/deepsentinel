import os, yaml

CONFIG = dict(
    
    pretrain ='VAE',                      # one of [None,'load',VAE','contrastive_loss','tile2vec']
    encoder = 'basic_bottleneck',         # one of ['basic_bottleneck']
    finetune = 'synthetic_rgb',                      # one of [None,'synthetic_rgb','landcover']
    load_run = None,
    sacred=dict(
        gcp_bucket='deepsentinel-models',
        gcp_basedir='/',
        local=os.path.join(os.getcwd(),'experiments','sacred'),
    ),
    encoder_params = dict(
        basic_bottleneck = dict(
            input_channels=5,
        ),
    ),
    pretrain_config=dict(
        VAE = dict(
            LR=0.00005,
            BATCH_SIZE=128,
            DATALOADER_WORKERS=6,
            EPOCHS=20,
            LOG_INTERVAL=10,
        ),
    ),
    pretrain_model_config=dict(
        VAE=dict(
            image_channels=5, 
            h_dim=int(((128/2/2/2/2-2)**2)*256), 
            z_dim=32,
        ),
    ), 
    pretrain_loader_config=dict(
        VAE=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), 
            data_dir='/data/DEMO_unlabelled/', 
            bands=['B2','B3','B4','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
    ),
    finetune_config=dict(
        synthetic_rgb=dict(
            LR=0.00005,
            EPOCHS=10,
            BATCH_SIZE=64,
            DATALOADER_WORKERS=6,
            LOG_INTERVAL=4,
        ),
    ), 
    finetune_model_config=dict(
        synthetic_rgb=dict(
            image_channels=5, 
            h_dim=int(((128/2/2/2/2-2)**2)*256), 
            z_dim=32,
        ),
    ), 
    finetune_loader_config=dict(
        synthetic_rgb=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), 
            data_dir='/data/DEMO_unlabelled/', 
            bands=['B2','B3','B4','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
    ), 
    encoder_layers=dict(
        basic_bottleneck=['encoder.net.0.weight', 'encoder.net.0.bias', 'encoder.net.2.weight', 'encoder.net.2.bias', 'encoder.net.4.weight', 'encoder.net.4.bias', 'encoder.net.6.weight', 'encoder.net.6.bias'],
    ), 
    device='cuda',
    verbose=True,
    vis_params = dict(
        VAE=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
            
        ),
        synthetic_rgb=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
            SYNTH_RGB=['VV','VV','VH'],
        )
    ),
    
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'conf','ML_CONFIG.yaml'),'w'))

