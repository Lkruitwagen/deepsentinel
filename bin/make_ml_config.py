import os, yaml

CONFIG = dict(
    
    model_spec='VAE',
    pretrain =True, 
    model_config=dict(
        VAE=dict(
            
        ),
    ), 
    pretrain_config=dict(
        VAE=dict(
        
        ),
    ), 
    encoder_layers=dict(
        VAE=dict(
        
        ),
    ), 
    device=0
    
)

yaml.dump(CONFIG, open(os.path.join(os.getcwd(),'ML_CONFIG.yaml'),'w'))