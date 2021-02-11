import os, yaml

CONFIG = dict(
    
    pretrain =None,                      # one of [None,'load',VAE','contrastive_loss','tile2vec']
    encoder = 'resnet18',         # one of ['basic_bottleneck','resnet18','resnet34','resnet50']
    finetune = 'landcover',                      # one of [None,'synthetic_rgb','landcover']
    load_run = None,
    pretraining_runs = dict(
        VAE = dict(
            resnet18=26
        ),
        tile2vec = dict(
            resnet18=69
        ),
        contrastive_loss = dict(
            resnet18=142
        )
    ),
    #finetune_synthrgb = dict(
    #    VAE = dict(
    #        resnet18='todo'
    #    ),
    #    tile2vec = dict(
    #        resnet18=84
    #    ),
    #),
    finetune_lc = dict(
        VAE = dict(
            resnet18=95
        ),
        tile2vec = dict(
            resnet18=94
        )
    
    )
    
    
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'conf','TEST_CONFIG.yaml'),'w'))

