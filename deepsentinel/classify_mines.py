import os, logging, json
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from shapely import geometry, wkt
import geopandas as gpd
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.exp import ex, EX_NAME
from deepsentinel.models.models import models
from deepsentinel.models.encoders import encoders
from deepsentinel.train import pretrain_vae, finetune_synthrgb, finetune_lc
from deepsentinel.dataloaders import dataloaders

logger = logging.getLogger('MINES-FINAL')
logging.basicConfig(level=logging.INFO)


def finetune(model, 
             finetune_loader_trn, 
             finetune_loader_val,
             optimizer, 
             finetune_params, 
             channel_stats,
             device, 
             verbose):
    
    n_iter = 0
    if not verbose:
        epoch_pbar = tqdm(total = pretrain_params['EPOCHS'])
        epoch_pbar.set_description(f'Epochs')
    
    if channel_stats:
        channel_stats = json.load(open(channel_stats,'r'))
    
    for epoch in range(1, finetune_params['EPOCHS'] + 1):
        
        trn_loss = 0
        if verbose:
            epoch_pbar = tqdm(total = len(finetune_loader_trn.dataset) + len(finetune_loader_val.dataset),ncols=100)
            epoch_pbar.set_description(f'Epoch {epoch}')
        
        Y_list = np.array([])
        Y_hat_list = np.array([])
        ### do training
        for batch_idx, (X, Y) in enumerate(finetune_loader_trn):
            n_iter +=1
            model.train()
            X, Y = X.to(device), Y.to(device)
            #print ('X,Y shape',X.shape, Y.shape)
            
            optimizer.zero_grad()
            Y_hat = model(X)
            loss = F.binary_cross_entropy(Y_hat, Y)
            loss.backward()
            optimizer.step()

            if (batch_idx % finetune_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                if verbose:
                    n_sample = (batch_idx * finetune_params['BATCH_SIZE'])
                    trn_loss+=loss.detach().item()
                    Y_list = np.concatenate([Y_list,np.squeeze(Y.cpu().detach().numpy())])
                    Y_hat_list = np.concatenate([Y_hat_list,np.squeeze(Y_hat.cpu().detach().numpy())])
                    TP = ((Y_list==(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
                    FP = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==0)).sum()
                    FN = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
                    desc = f'Epoch {epoch} - N:{Y_list.sum()}, Pr: {safediv(TP,TP+FP):.2f}, Re: {safediv(TP,TP+FN):.2f}, avgtrnloss {safediv(trn_loss,n_sample):.6f} - avgvalloss nan'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(X.shape[0]*finetune_params['LOG_INTERVAL'])
        
        trn_pr = TP/(TP+FP)
        trn_re = TP/(TP+FN)
        
        del loss        
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        
        ### do cross-validation
        Y_list = np.array([])
        Y_hat_list = np.array([])
        val_loss=0
        model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(finetune_loader_val):
                n_iter +=1
                X, Y = X.to(device), Y.to(device)
                Y_hat = model(X)
                #print ('Y_hat shape',Y_hat.shape)

                loss = F.binary_cross_entropy(Y_hat, Y)
                val_loss+= loss.detach().item()

                if (batch_idx % finetune_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                    if verbose:
                        n_sample = (batch_idx * finetune_params['BATCH_SIZE'])
                        val_loss+=loss.detach().item()
                        Y_list = np.concatenate([Y_list,np.squeeze(Y.cpu().detach().numpy())])
                        Y_hat_list = np.concatenate([Y_hat_list,np.squeeze(Y_hat.cpu().detach().numpy())])
                        TP = ((Y_list==(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
                        FP = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==0)).sum()
                        FN = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
                        desc = f'Epoch {epoch} - Pr: {trn_pr:.2f}, Re: {trn_re:.2f}, avgtrnloss {safediv(trn_loss,n_sample):.6f} -  N:{Y_list.sum()}, Pr: {safediv(TP,TP+FP):.2f}, Re: {safediv(TP,TP+FN):.2f}, avgvalloss {safediv(val_loss,n_sample):.6f}'
                        epoch_pbar.set_description(desc)
                        epoch_pbar.update(X.shape[0]*finetune_params['LOG_INTERVAL'])
                    
        if verbose:
            desc = f'Epoch {epoch} - Pr: {trn_pr:.2f}, Re: {trn_re:.2f}, avgtrnloss {safediv(trn_loss,n_sample):.6f} -  N:{Y_list.sum()}, Pr: {safediv(TP,TP+FP):.2f}, Re: {safediv(TP,TP+FN):.2f}, avgvalloss {safediv(val_loss,n_sample):.6f}'
            epoch_pbar.set_description(desc)
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
            
        del loss
        
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

def safediv(numer,denom):
    if denom==0:
        return np.nan
    else:
        return numer/denom

def classify_mines(mines_load_run, encoder, encoder_params, mines_model_config, mines_config, mines_loader_config, vis_params, device, verbose, random_seed, trn_split, cv_split, NE_ROOT, **kwargs):
    """
    The main loop that includes pretraining, training, testing, and i/o for tensorboard and Sacred
    
    Returns
    -------
        None
    """
    logger = logging.getLogger('Classify Mines')
    logger.info('Initialising model and optimizer')
    # initialise the dataloaders
    
    
    ## load the model for finetuning
    model = models['mining'](encoder, encoder_params[encoder], **mines_model_config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=mines_config['LR'])
    
    logger.info('Loading saved run')
    finetuned_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(mines_load_run), 'finetuned_model.pth'))
    # filter the keys needed for the encoder
    finetuned_dict = {kk:vv for kk,vv in finetuned_dict.items() if 'encoder' in kk}
    # get the state_dict for the finetuning model
    model_dict = model.state_dict()
    # update the state dict
    model_dict.update(finetuned_dict)
    # load the updated model_dict
    model.load_state_dict(model_dict)
    

    
    #### Finetune
    
    logger.info('Setting up dataloaders')
    mines_loader_trn = DataLoader(
            dataloaders['mines'](seed=random_seed, start_por=0., end_por=trn_split, data_dir='/data/mines_labelled/',label_cols=['COAL'],inference=False, **mines_loader_config), 
            batch_size=mines_config['BATCH_SIZE'], 
            shuffle=True,
            num_workers=mines_config['DATALOADER_WORKERS']
        )
    mines_loader_val = DataLoader(
            dataloaders['mines'](seed=random_seed, start_por=trn_split, end_por=cv_split, data_dir='/data/mines_labelled/',label_cols=['COAL'],inference=False,  **mines_loader_config), 
            batch_size=mines_config['BATCH_SIZE'], 
            shuffle=True,
            num_workers=mines_config['DATALOADER_WORKERS']
        )
    mines_loader_test = DataLoader(
            dataloaders['mines'](seed=random_seed, start_por=cv_split, end_por=1., data_dir='/data/mines_labelled/',label_cols=['COAL'],inference=False,   **mines_loader_config), 
            batch_size=mines_config['BATCH_SIZE'], 
            shuffle=True,
            num_workers=mines_config['DATALOADER_WORKERS']
        )
    
    logger.info('Final finetuning')
    finetune(model, 
             mines_loader_trn, 
             mines_loader_val,
             optimizer, 
             mines_config, 
             mines_loader_config['channel_stats'],
             device, 
             verbose)
    
    
    #### report final test accuracy
    logger.info('Test accuracy')
    Y_list = np.array([])
    Y_hat_list = np.array([])
    test_loss=0
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, Y) in enumerate(mines_loader_test):
            X, Y = X.to(device), Y.to(device)
            Y_hat = model(X)
            #print ('Y_hat shape',Y_hat.shape)
            
            loss = F.binary_cross_entropy(Y_hat, Y)           
            n_sample = (batch_idx * mines_config['BATCH_SIZE'])
            test_loss+=loss.detach().item()

            Y_list = np.concatenate([Y_list,np.squeeze(Y.cpu().detach().numpy())])
            Y_hat_list = np.concatenate([Y_hat_list,np.squeeze(Y_hat.cpu().detach().numpy())])
            TP = ((Y_list==(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
            FP = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==0)).sum()
            FN = ((Y_list!=(Y_hat_list>0.5).astype(int)) & (Y_list==1)).sum()
            desc = f'N:{Y_list.sum()}, Pr: {safediv(TP,TP+FP):.2f}, Re: {safediv(TP,TP+FN):.2f}, avgtestloss {safediv(test_loss,n_sample):.6f}'
            logger.info(desc)

        
    auc_roc = roc_auc_score(Y_list, Y_hat_list)    
    logger.info(f'AUC_ROC: {auc_roc:.3f}')
    with open(f'./logs/{mines_load_run}_metrics.log','a') as f:
        f.write(f'AUC_ROC: {auc_roc:.3f}')
    
    
    ## run inference on the full dataset
    logger.info('Running full inference')
    mines_inference_loader = DataLoader(
            dataloaders['mines'](seed=random_seed, start_por=0., end_por=1., data_dir='/data/mines_unlabelled/',label_cols=[],inference=False, **mines_loader_config), 
            batch_size=mines_config['BATCH_SIZE'], 
            shuffle=True,
            num_workers=mines_config['DATALOADER_WORKERS']
        )
    
    
    Y_hat_list = np.array([])
    test_loss=0
    model.eval()
    with torch.no_grad():
        for batch_idx, (X,_) in enumerate(mines_inference_loader):
            X = X.to(device)
            Y_hat = model(X)
            Y_hat_list = np.concatenate([Y_hat_list, np.squeeze(Y_hat.cpu().detach().numpy())])
    
    logger.info('post-processing')
    
    Y_hat_df = pd.DataFrame(zip([r['record'] for r in mines_inference_loader.dataset.records] , Y_hat_list.tolist()), columns=['idx','Y_hat'])
    print ('pts')
    print (mines_inference_loader.dataset.pts)
    pd.merge(mines_inference_loader.dataset.pts, Y_hat_df, how='left',left_index=True, right_on='idx').to_parquet(os.path.join(NE_ROOT,'pts','mines_inferred.parquet'))
    print (pd.merge(mines_inference_loader.dataset.pts, Y_hat_df, how='left',left_index=True, right_on='idx'))
    
    
def mines_postprocess():
    
    
    # move back to gdf. Combine with maus -> spatial join on bbox_wgs, left. intersect geom, intersect_geom area, groupby left mean.
    gdf = pd.read_parquet(os.path.join(CONFIG['NE_ROOT'],'pts','mines_inferred.parquet'))
    gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(gdf,geometry='geometry')
    
    
    maus_mines = gpd.read_file(os.path.join(CONFIG['NE_ROOT'],'pts','maus.gpkg'))
    nrgi_mines = gpd.read_file(os.path.join(CONFIG['NE_ROOT'],'pts','ngri.gpkg'))
    
    # finish later, test in ipynb first
    
    