import os, yaml

CONFIG = dict(
    NE_ROOT = os.path.join(os.getcwd(),'data'),
    pretrain ='contrastive_loss',                      # one of [None,'load','VAE','tile2vec','contrastive_loss']
    encoder = 'resnet18',         # one of ['basic_bottleneck','resnet18','resnet34','resnet50']
    finetune = 'landcover',                      # one of [None,'synthetic_rgb','landcover']
    load_run = None,
    sacred=dict(
        gcp_bucket='deepsentinel-models',
        gcp_basedir='',
        local=os.path.join(os.getcwd(),'experiments','sacred'),
    ),
    encoder_params = dict(
        basic_bottleneck = dict(
            input_channels=5,
        ),
        resnet18 = dict(
            input_channels=14,
        ),
    ),
    pretrain_config=dict(
        VAE = dict(
            LR=0.00005,
            BATCH_SIZE=600,
            DATALOADER_WORKERS=24,
            EPOCHS=20,
            LOG_INTERVAL=10,
            LOSS_CONVERGENCE=0.0005,
            EPOCH_BREAK_WINDOW=50,
        ),
        tile2vec = dict(
            LR=0.00005,
            BATCH_SIZE=600,#600,
            DATALOADER_WORKERS=24,
            EPOCHS=20,
            LOG_INTERVAL=10,
            LOSS_CONVERGENCE=0.0005,
            EPOCH_BREAK_WINDOW=50,
        ),
        contrastive_loss = dict(
            LR=0.00005,
            BATCH_SIZE=768,#600,
            DATALOADER_WORKERS=24,
            EPOCHS=1,
            LOG_INTERVAL=1,
            LOSS_CONVERGENCE=0.0005,
            EPOCH_BREAK_WINDOW=50,
        ),
    ),
    pretrain_model_config=dict(
        VAE=dict(
            image_channels=14, 
            c_dim=64, # channels in pixel representation before normal transform
            h_dim=64*8*8, # driven by encoder output
            z_dim=64*8*8,
            bilinear=False,
        ),
        tile2vec=dict(
            image_channels=14, 
            bilinear=False,
        ),
        contrastive_loss = dict(
            activation='relu',
        )
    ), 
    pretrain_loader_config=dict(
        VAE=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), 
            data_dir='/data_100/100k_unlabelled/', 
            bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
        tile2vec=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), 
            data_dir='/data_100/100k_unlabelled/', 
            bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
        contrastive_loss=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_CONFIG.yaml'), 
            data_dir='/data_100/100k_unlabelled/', 
            bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128,
            augmentations=['crop','dropout','jitter'],
            random_crop = True,
            warmup_epochs = 5,
            ramp_epochs = 10,
            s2_dropout = {'max':0.3, 'min':0.01},
            s1_dropout = {'max':0.3, 'min':0.01},
            N_jitters = {'max':10,'min':1},
            aug_crop={'max':16,'min':1},
            jitter_params = dict(
                brightness = {'max':0.1,'min':0.01},
                contrast = {'max':0.1,'min':0.01},
                saturation = {'max':0.1,'min':0.01},
                hue = {'max':0.1,'min':0.01},
            )
        ),
    ),
    finetune_config=dict(
        synthetic_rgb=dict(
            LR=0.00005,
            EPOCHS=150,
            BATCH_SIZE=512,
            DATALOADER_WORKERS=24,
            LOG_INTERVAL=1,
            EPOCH_BREAK_WINDOW=100
        ),
        landcover=dict(
            LR=0.0005,
            EPOCHS=200,
            DATALOADER_WORKERS=24,
            BATCH_SIZE=600,
            LOG_INTERVAL=1,
            EPOCH_BREAK_WINDOW=40,
        )
    ), 
    finetune_model_config=dict(
        synthetic_rgb=dict(
            bilinear=False,
            activation=None,
        ),
        landcover = dict(
            bilinear=False,
            activation='sigmoid'
        )
    ), 
    finetune_loader_config=dict(
        synthetic_rgb=dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_100_CONFIG.yaml'), 
            data_dir='/data_100/10k_labelled/', 
            bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH'], 
            source='GEE', 
            channel_stats = os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'), 
            patch_size=128
        ),
        landcover = dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_100_CONFIG.yaml'),
            data_dir='/data_100/10k_labelled/',
            bands=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','VV','VH'],
            source='GEE',
            channel_stats=os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'),
            patch_size=128
        )
    ), 
    encoder_layers=dict(
        basic_bottleneck=['encoder.net.0.weight', 'encoder.net.0.bias', 'encoder.net.2.weight', 'encoder.net.2.bias', 'encoder.net.4.weight', 'encoder.net.4.bias', 'encoder.net.6.weight', 'encoder.net.6.bias'],
        resnet18 = [],
    ), 
    device='cuda',
    verbose=True,
    vis_params = dict(
        VAE=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
        ),
        tile2vec=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
        ),
        synthetic_rgb=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
            SYNTH_RGB=['VV','VV','VH'],
        ),
        landcover=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
            SYNTH_RGB=['VV','VV','VH'],
        ),
        contrastive_loss=dict(
            IMAGE_SAMPLES=10,
            RGB_BANDS=['B4','B3','B2'],
            SYNTH_RGB=['VV','VV','VH'],
        )
    ),
    random_seed = 5,
    trn_split= 0.7,
    cv_split = 0.9,
    mines_load_run = 179,
    mines_model_config = dict(
        n_classes=1,
        bilinear=False,
    ), 
    mines_config = dict(
            LR=0.00001,
            EPOCHS=5,
            BATCH_SIZE=256,
            DATALOADER_WORKERS=6,
            LOG_INTERVAL=4,
    ),
    mines_loader_config = dict(
            data_config=os.path.join(os.getcwd(),'conf','DATA_100_CONFIG.yaml'),
            bands=['B2','B3','B4','VV','VH'],
            source='GEE',
            channel_stats=os.path.join(os.getcwd(),'data','channel_stats','DEMO_unlabelled_GEE.json'),
            patch_size=128
    ),
    
    
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'conf','ML_CONFIG.yaml'),'w'))

