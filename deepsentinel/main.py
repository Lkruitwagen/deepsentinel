import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.exp import ex, EX_NAME
from deepsentinel.models.models import models
from deepsentinel.models.encoders import encoders
from deepsentinel.train import pretrain_loop, finetune_synthrgb, finetune_lc, test
from deepsentinel.dataloaders import dataloaders

"""A script to instantiate the experiment object"""
import os, glob, yaml
from sacred import Experiment

# generate NAME
### todo: get from globbing experiments dir
EX_NAME='test'
ex = Experiment(EX_NAME)


@ex.automain
def main(pretrain, finetune, load_run, encoder, encoder_params, encoder_layers, pretrain_config, pretrain_model_config, pretrain_loader_config,finetune_config, finetune_model_config, finetune_loader_config, vis_params, device, verbose, _log, _run):
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
    
    
    if pretrain!=None and pretrain!='load':
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
            num_workers=pretrain_config[pretrain]['DATALOADER_WORKERS']
        )

        _log.info('Call the pretraining')
        pretrain_loop(pretrain_model, 
                 pretrain_loader, 
                 optimizer, 
                 writer,
                 pretrain_config[pretrain],  
                 pretrain_loader_config[pretrain]['channel_stats'],
                 vis_params[pretrain],
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
        _log.info('~~~FINETUNING~~~')
        _log.info(f'Finetuning for {finetune}')
        
        finetune_model = models[finetune](encoder, encoder_params[encoder], **finetune_model_config[finetune]).to(device)
        
        if load_run and pretrain=='load':
            _log.info(f'Loading pretained encoder {load_run}')
            
            # load a previous pretrained model
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(load_run), 'pretrained_model.pth'))
            
        else:
            _log.info(f'Loading pretained encoder {_run._id}')
            
            # load this run's pretrained model
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(_run._id), 'pretrained_model.pth'))
            
        # filter the keys needed for the encoder
        pretrained_dict = {kk:vv for kk,vv in pretrained_dict.items() if kk in encoder_layers[encoder]}
        # get the state_dict for the finetuning model
        model_dict = finetune_model.state_dict()
        # update the state dict
        model_dict.update(pretrained_dict)
        # load the updated model_dict
        finetune_model.load_state_dict(model_dict)
    
        _log.info('instantiate the optimizer')
        optimizer = torch.optim.Adam(params=finetune_model.parameters(), lr=finetune_config[finetune]['LR'])
        
        _log.info('instantiate the dataloader')
        finetune_dataset = dataloaders[finetune](**finetune_loader_config[finetune])
        
        finetune_loader = DataLoader(
            finetune_dataset, 
            batch_size=finetune_config[finetune]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=finetune_config[finetune]['DATALOADER_WORKERS'], 
        )

        _log.info('Call the finetuning') # can abstract this if necessary
        if finetune=='synthetic_rgb':
            print ('device',device)
            print ('verbose',verbose)
            finetune_synthrgb(finetune_model, 
                     finetune_loader, 
                     optimizer, 
                     writer,
                     finetune_config[finetune],  
                     finetune_loader_config[finetune]['channel_stats'],
                     vis_params[finetune],
                     device, 
                     verbose)
        elif finetune=='landcover':
            finetune_lc(finetune_model, 
                     finetune_loader, 
                     optimizer, 
                     writer,
                     finetune_config[finetune],  
                     finetune_loader_config[finetune]['channel_stats'],
                     vis_params[finetune],
                     device, 
                     verbose)
        
        _log.info('save the model')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_model.pth'), name='finetuned_model.pth')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_optimizer.pth'), name='finetuned_optimizer.pth')
