import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.exp import ex, EX_NAME
from deepsentinel.models.models import models
from deepsentinel.models.encoders import encoders
from deepsentinel.train import pretrain_vae, finetune_synthrgb, finetune_lc, pretrain_t2v, pretrain_csf
from deepsentinel.dataloaders import dataloaders

"""A script to instantiate the experiment object"""
import os, glob, yaml
from sacred import Experiment

# generate NAME
### todo: get from globbing experiments dir
EX_NAME='test'
ex = Experiment(EX_NAME)


@ex.automain
def main(pretrain, finetune, load_run, encoder, encoder_params, encoder_layers, pretrain_config, pretrain_model_config, pretrain_loader_config,finetune_config, finetune_model_config, finetune_loader_config, vis_params, device, verbose, random_seed, trn_split, cv_split, _log, _run):
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
        
        pretrain_model = models[pretrain](encoder, encoder_params[encoder], **pretrain_model_config[pretrain])
        
        #if verbose:
        #    _log.info('Model Summary')
        #    print (summary(pretrain_model, input_size=(5, 128, 128), device='cuda'))
        
        if pretrain!='aegan':
            if torch.cuda.device_count() > 1:
                _log.info(f"let's use {torch.cuda.device_count()} GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                pretrain_model = torch.nn.DataParallel(pretrain_model)

            pretrain_model = pretrain_model.to(device)
            _log.info('~~~PRETRAINING~~~')
            _log.info('Instantiate the optimizer')
            optimizer = torch.optim.Adam(params=pretrain_model.parameters(), lr=pretrain_config[pretrain]['LR'])
        else:
            if torch.cuda.device_count() > 1:
                _log.info(f"let's use {torch.cuda.device_count()} GPUs!")
                for kk in pretrain_model.keys():
                    pretrain_model[kk] = torch.nn.DataParallel(pretrain_model[kk])
            for kk in pretrain_model.keys():
                pretrain_model[kk] = pretrain_model[kk].to(device)
            _log.info('~~~PRETRAINING~~~')
            _log.info('Instantiate the optimizer')
            optimizers = {}
            for kk in pretrain_model.keys():
                optimizers[kk] = torch.optim.Adam(params=pretrain_model[kk].parameters(), lr=pretrain_config[pretrain]['LR'])
            optimizers['encoder-GAN'] = torch.optim.Adam(params=pretrain_model['encoder'].parameters(), lr=pretrain_config[pretrain]['LR'])
            optimizers['decoder-GAN'] = torch.optim.Adam(params=pretrain_model['decoder'].parameters(), lr=pretrain_config[pretrain]['LR'])
            
        
        _log.info('Instantiate the Dataloader')
        pretrain_dataset = dataloaders[pretrain](**pretrain_loader_config[pretrain])
        
        pretrain_loader = DataLoader(
            pretrain_dataset, 
            batch_size=pretrain_config[pretrain]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=pretrain_config[pretrain]['DATALOADER_WORKERS']
        )

        _log.info('Call the pretraining')
        if pretrain=='VAE':
            pretrain_vae(pretrain_model, 
                     pretrain_loader, 
                     optimizer, 
                     writer,
                     pretrain_config[pretrain],  
                     pretrain_loader_config[pretrain]['channel_stats'],
                     vis_params[pretrain],
                     device, 
                     verbose)
        elif pretrain=='tile2vec':
            pretrain_t2v(pretrain_model, 
                     pretrain_loader, 
                     optimizer, 
                     writer,
                     pretrain_config[pretrain],  
                     pretrain_loader_config[pretrain]['channel_stats'],
                     vis_params[pretrain],
                     device, 
                     verbose)
        elif pretrain=='contrastive_loss':
            pretrain_csf(pretrain_model, 
                     pretrain_loader, 
                     optimizer, 
                     writer,
                     pretrain_config[pretrain],  
                     pretrain_loader_config[pretrain]['channel_stats'],
                     vis_params[pretrain],
                     device, 
                     verbose)
        elif pretrain=='AEGAN':
            pretrain_aegan(pretrain_model,  #dict
                     pretrain_loader, 
                     optimizer, #dict
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
        
        # do the loader first, then get the model parameters from the loader
        _log.info('instantiate the dataloader')
        finetune_dataset = dataloaders[finetune](**finetune_loader_config[finetune])
        
        finetune_loader_trn = DataLoader(
            dataloaders[finetune](seed=random_seed, start_por=0., end_por=trn_split,**finetune_loader_config[finetune]), 
            batch_size=finetune_config[finetune]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=finetune_config[finetune]['DATALOADER_WORKERS'], 
        )
        
        finetune_loader_val = DataLoader(
            dataloaders[finetune](seed=random_seed, start_por=trn_split, end_por=cv_split,**finetune_loader_config[finetune]), 
            batch_size=finetune_config[finetune]['BATCH_SIZE'], 
            shuffle=False,
            num_workers=finetune_config[finetune]['DATALOADER_WORKERS'], 
        )
        
        _, _Y = next(iter(finetune_loader_trn))
        print ('n classes', _Y.shape[1])
        
        finetune_model = models[finetune](encoder, encoder_params[encoder], n_classes=_Y.shape[1], **finetune_model_config[finetune])
        _log.info('instantiate the optimizer')
        optimizer = torch.optim.Adam(params=finetune_model.parameters(), lr=finetune_config[finetune]['LR'])
        

        
        #for name, layer in finetune_model.named_modules():
        #    print(name)
        
        #if verbose:
        #    _log.info('Finetune Model Summary')
        #    print (summary(finetune_model, input_size=(5, 128, 128)))
        
        if load_run and pretrain=='load':
            _log.info(f'Loading pretained encoder {load_run}')
            
            # load a previous pretrained model
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(load_run), 'pretrained_model.pth'), map_location='cpu')
            optimizer_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(load_run), 'pretrained_optimizer.pth'), map_location='cpu')
            
            # filter the keys needed for the encoder
            pretrained_dict = {kk:vv for kk,vv in pretrained_dict.items() if 'encoder' in kk}
            print ('pretrained test', pretrained_dict['encoder.conv1.weight'].mean())
            #print ('pretrained keys')
            #print (pretrained_dict.keys())
            # get the state_dict for the finetuning model
            model_dict = finetune_model.state_dict()
            print ('modeldict test', model_dict['encoder.conv1.weight'].mean())
            #print ('finetune keys')
            #print (model_dict.keys())
            print ('overlap keys')
            print (len([kk for kk in pretrained_dict.keys() if kk in model_dict.keys()]))
            # update the state dict
            #model_dict.update(pretrained_dict)
            # load the updated model_dict
            finetune_model.load_state_dict(pretrained_dict, strict=False) # put on cpu first then move to GPU
            optimizer.load_state_dict(optimizer_dict, strict=False) # put on cpu first then move to GPU
            model_dict = finetune_model.state_dict()
            print ('modeldict test - post', model_dict['encoder.conv1.weight'].mean())
            
        elif pretrain !=None:
            _log.info(f'Loading pretained encoder {_run._id}')
            
            # load this run's pretrained model
            pretrained_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(_run._id), 'pretrained_model.pth'), map_location='cpu')
            optimizer_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(_run._id), 'pretrained_optimizer.pth'), map_location='cpu')
            
            # filter the keys needed for the encoder
            pretrained_dict = {kk:vv for kk,vv in pretrained_dict.items() if 'encoder' in kk}
            # get the state_dict for the finetuning model
            model_dict = finetune_model.state_dict()
            # update the state dict
            model_dict.update(pretrained_dict)
            # load the updated model_dict
            finetune_model.load_state_dict(pretrained_dict, strict=False)
            optimizer.load_state_dict(optimizer_dict, strict=False) # put on cpu first then move to GPU
            
        _log.info('optionally parallelise model and move to gpu(s)')
        if torch.cuda.device_count() > 1:
            _log.info(f"let's use {torch.cuda.device_count()} GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            finetune_model = torch.nn.DataParallel(finetune_model)
        
        finetune_model.to(device)
        optimizer.to(device)
    
        
        
        

        _log.info('Call the finetuning') # can abstract this if necessary
        if finetune=='synthetic_rgb':
            print ('device',device)
            print ('verbose',verbose)
            finetune_synthrgb(finetune_model, 
                     finetune_loader_trn,
                     finetune_loader_val,
                     optimizer, 
                     writer,
                     finetune_config[finetune],  
                     finetune_loader_config[finetune]['channel_stats'],
                     vis_params[finetune],
                     device, 
                     verbose)
        elif finetune=='landcover':
            finetune_lc(finetune_model, 
                     finetune_loader_trn,
                     finetune_loader_val,
                     optimizer, 
                     writer,
                     finetune_config[finetune],  
                     finetune_loader_config[finetune]['channel_stats'],
                     vis_params[finetune],
                     device, 
                     verbose)
        
        _log.info('save the model')
        ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_model.pth'), name='finetuned_model.pth')
        #ex.add_artifact(filename=os.path.join(os.getcwd(), 'tmp', 'finetune_optimizer.pth'), name='finetuned_optimizer.pth')
