import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.exp import ex, EX_NAME
from deepsentinel.models.models import models
from deepsentinel.models.encoders import encoders
from deepsentinel.train import pretrain_loop, train_synthrgb, train_lc, test
from deepsentinel.dataloaders import dataloaders

"""A script to instantiate the experiment object"""
import os, glob, yaml
from sacred import Experiment

# generate NAME
### todo: get from globbing experiments dir
EX_NAME='test'
ex = Experiment(EX_NAME)


@ex.automain
def main(pretrain, finetune, load_run, pretrain_config, pretrain_model_config, encoder, encoder_params, pretrain_loader_config, encoder_layers, device, verbose, _log, _run):
    """
    The main loop that includes pretraining, training, testing, and i/o for tensorboard and Sacred
    
    Returns
    -------
        None
    """

    _log.info('Setting up writer')
    # initialise the dataloaders
    writer = SummaryWriter(os.path.join(os.getcwd(), 'experiments', 'tensorboard',_run._id))
    base_sacred_path = os.path.join(os.getcwd(), 'experiments', 'sacred', _run._id)
    
    
    if pretrain:
        _log.info('Initialising pretrain model and sending to device')
        
        pretrain_model = models[pretrain](encoder, encoder_params[encoder], **pretrain_model_config[pretrain]).to(device)
        if verbose:
            _log.info('Model Summary')
            print (summary(pretrain_model, input_size=(5, 128, 128)))
            
        _log.info('~~~PRETRAINING~~~')
        _log.info('Instantiate the optimizer')
        optimizer = torch.optim.Adam(params=pretrain_model.parameters(), lr=pretrain_config[pretrain]['LR'])
        
        _log.info('Instantiate the Dataloader')
        pretrain_dataset = dataloaders[pretrain](**pretrain_loader_config[pretrain])
        
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=pretrain_config[pretrain]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=pretrain_config[pretrain]['DATALOADER_WORKERS'], 
        )

        _log.info('Call the pretraining')
        pretrain_loop(pretrain_model, 
                 pretrain_loader, 
                 optimizer, 
                 writer,
                 pretrain_config[pretrain],  
                 device, 
                 verbose)
        
        _log.info('save the model')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'pretrain_model.pth'), name='pretrained_model.pth')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'pretrain_optimizer.pth'), name='pretrained_optimizer.pth')
        
        del pretrain_model
        del optimizer
    
    else:
        _log.info('Skipping pretraining.')    
            
    
    if finetune:
        _log.info(f'Finetuning for {finetune}')
        
        finetune_model = models[finetune](**finetune_config[finetune]).to(device)
        
        if load_run:
            _log.info(f'Loading pretained encoder {load_run}')
            # do the specific layers of the model
            finetune_model.load_state_dict(torch.load(os.path.join(base_sacred_path, 'pretrained_model.pth')))
            #optimizer.load_state_dict(torch.load(os.path.join(base_sacred_path, load_run,'pretrained_optimizer.pth')))
    
        _log.info('~~~FINETUNING~~~')
        _log.info('instantiate the optimizer')
        optimizer = torch.optim.Adam(params=pretrain_model.parameters(), lr=finetune_config[finetune]['LR'])
        
        _log.info('instantiate the dataloader')
        finetune_dataset = dataloaders[finetune](**pretrain_loader_config[finetune])
        
        finetune_loader = DataLoader(
            finetune_dataset, 
            batch_size=finetune_config[finetune]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=finetune_config[finetune]['DATALOADER_WORKERS'], 
        )

        _log.info('Call the finetuning')
        finetune_loop(finetune_model, 
                 finetune_loader, 
                 optimizer, 
                 writer,
                 finetune_config[finetune],  
                 device, 
                 verbose)
        
        _log.info('save the model')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_model.pth'), name='finetuned_model.pth')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_optimizer.pth'), name='finetuned_optimizer.pth')
        
        
            
            

    
    
    
    
    
    
    
    # 64  -> 2x2
    # 96  -> 4x4
    # 128 -> 6x6
    # patch_size/2/2/2/2-2 -> final_patch; final_patch**2 * 256

        
    ## obtain the (now) pre-trained encoder layers
    
    """
    pretrained_dict = ...
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_layers[model_spec]}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    encoder = model
    """
