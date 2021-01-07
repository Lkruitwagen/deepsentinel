import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.exp import ex, EX_NAME
from deepsentinel.encoders import models
from deepsentinel.train import pretrain_loop, train, test
from deepsentinel.dataloaders import dataloaders

@ex.automain
def main(pretrain, model_spec, model_config, pretrain_config, pretrain_loader_config, encoder_layers, device, verbose, _log, _run):
    """
    The main loop that includes pretraining, training, testing, and i/o for tensorboard and Sacred
    
    Parameters
    ----------
        pretrain: bool
            Whether to pretrain the encoder or use a previously trained one
        model_spec: str
            Which encoder architecture to use
        model_config: dict
            Config vars for the encoder
        pretrain_config: dict
            Config vars for the pretraining
        encoder_layers: dict
            Dict keyed by model_spec with a list of layers to retain as the encoder
        device: str 
            The device name to use for training
    
    Returns
    -------
        None
    """

    _log.info('Setting up writer')
    # initialise the dataloaders
    writer = SummaryWriter(os.path.join(os.getcwd(), 'experiments', 'tensorboard',_run._id))
    base_sacred_path = os.path.join(os.getcwd(), 'experiments', 'sacred', _run._id)

    
    _log.info('Initialising pretrain model and sending to device')
    pretrain_model = models[model_spec](**model_config[model_spec]).to(device)
    
    if verbose:
        _log.info('Model Summary')
        print (summary(pretrain_model, input_size=(5, 128, 128)))
    optimizer = torch.optim.Adam(params=pretrain_model.parameters(), lr=pretrain_config[model_spec]['LR'])
    
    
    
    # 64  -> 2x2
    # 96  -> 4x4
    # 128 -> 6x6
    # patch_size/2/2/2/2-2 -> final_patch; final_patch**2 * 256
    
    
    # run the self-supervised pretraining
    _log.info('~~~ PRETRAINING ~~~')
    if pretrain:
        ## do the pretraining
        
        # instantiate the dataloader
        _log.info('Instantiate the Dataloader')
        pretrain_dataset = dataloaders[model_spec](**pretrain_loader_config[model_spec])
        
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=pretrain_config[model_spec]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=pretrain_config[model_spec]['DATALOADER_WORKERS'], 
        )

        
        # .. do training
        _log.info('Call the pretraining')
        pretrain_loop(pretrain_model, 
                 pretrain_loader, 
                 optimizer, 
                 writer,
                 pretrain_config[model_spec],  
                 device, 
                 verbose)
        
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'pretrain_model.pth'), name='pretrained_model.pth')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'pretraoptimizer.pth'), name='pretrained_optimizer.pth')

        
    else:
        ## load a pre-trained model
        pretrain_model.load_state_dict(torch.load(os.path.join(base_sacred_path, 'pretrained_model.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(base_sacred_path, 'pretrained_optimizer.pth')))
        
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
    
    # run the supervised fine-tuning
    cv_loader = cv_loader()
    # ... do cv
    
    # test
    #... todo
    