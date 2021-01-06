import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepsentinel.exp import ex
from deepsentinel.encoders import models
from deepsentinel.train import pretrain, train, test
from deepsentinel.dataloaders import dataloaders

@ex.automain
def main(pretrain, model_spec, model_config, pretrain_config, encoder_layers, device):
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
    
    # initialise the dataloaders
    writer = SummaryWriter(os.path.join(os.getcwd(), 'experiments', 'tensorboard',ex.name))
    base_sacred_path = os.path.join(os.getcwd(), 'experiments', 'sacred', ex.name)
    pretrain_model = models[model_spec](model_config[model_spec]).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=pretrain_config['LR'])
    
    
    # run the self-supervised pretraining
    if pretain:
        ## do the pretraining
        
        # instantiate the dataloader
        pretrain_loader = dataloaders[model_spec](pretrain_config[model_spec])
        
        # .. do training
        pretrain(pretrain_model, 
                 pretrain_loader, 
                 optimizer, 
                 writer,
                 train_params, 
                 model_params, 
                 device, 
                 verbose=False)
        
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'model.pth'), name='saved_model.pth')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'optimizer.pth'), name='saved_optimizer.pth')

        
    else:
        ## load a pre-trained model
        pretrain_model.load_state_dict(torch.load(os.path.join(base_sacred_path, 'saved_model.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(base_sacred_path, 'saved_optimizer.pth')))
        
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
    