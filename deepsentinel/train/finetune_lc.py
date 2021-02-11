import os, json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsentinel.exp import ex
from deepsentinel.models.visualisation import plot_rgb, plot_categorical

@ex.capture
def finetune_lc(finetune_model, 
                 finetune_loader_trn, 
                 finetune_loader_val,
                 optimizer, 
                 writer,
                 finetune_params, 
                 channel_stats,
                 vis_params,
                 device, 
                 verbose):
    n_iter = 0
    if not verbose:
        epoch_pbar = tqdm(total = pretrain_params['EPOCHS'])
        epoch_pbar.set_description(f'Epochs')
    
    if channel_stats:
        channel_stats = json.load(open(channel_stats,'r'))
        
    ### loss function
    criterion = nn.CrossEntropyLoss()
    epoch_losses = np.array([])
    
    for epoch in range(1, finetune_params['EPOCHS'] + 1):
        
        trn_loss = 0
        if verbose:
            epoch_pbar = tqdm(total = len(finetune_loader_trn.dataset) + len(finetune_loader_val.dataset))
            epoch_pbar.set_description(f'Epoch {epoch}')
        
        ### do training
        for batch_idx, (X, Y) in enumerate(finetune_loader_trn):
            n_iter +=1
            finetune_model.train()
            X, Y = X.to(device), Y.to(device)
            #print ('X,Y shape',X.shape, Y.shape)
            
            optimizer.zero_grad()
            Y_hat = finetune_model(X)
            loss = criterion(Y_hat, torch.argmax(Y, dim=1))
            loss.backward()
            optimizer.step()

            if (batch_idx % finetune_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                if verbose:
                    n_sample = (batch_idx * finetune_params['BATCH_SIZE'])
                    trn_loss+=loss.detach().item()
                    desc = f'Epoch {epoch} - avgtrnloss {trn_loss/n_sample:.6f} - avgvalloss nan'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(X.shape[0]*finetune_params['LOG_INTERVAL'])
                    
        writer.add_scalar('Finetune_Loss/train', trn_loss, epoch)
        
        del loss        
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        
            
        ### do cross-validation
        val_loss=0
        finetune_model.eval()
        with torch.no_grad():
            for batch_idx, (X, Y) in enumerate(finetune_loader_val):
                n_iter +=1
                X, Y = X.to(device), Y.to(device)
                Y_hat = finetune_model(X)
                #print ('Y_hat shape',Y_hat.shape)

                loss = criterion(Y_hat, torch.argmax(Y, dim=1))
                val_loss+= loss.detach().item()

                if (batch_idx % finetune_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                    if verbose:
                        n_sample = (batch_idx * finetune_params['BATCH_SIZE'])
                        val_loss+=loss.detach().item()
                        desc = f'Epoch {epoch} - avgtrnloss {trn_loss/n_sample:.6f} - avgvalloss {val_loss/n_sample:.6f}'
                        epoch_pbar.set_description(desc)
                        epoch_pbar.update(X.shape[0]*finetune_params['LOG_INTERVAL'])
                    
        writer.add_scalar('Finetune_Loss/crossval', val_loss, epoch)
        
        epoch_loss = val_loss/len(finetune_loader_val.dataset)
        epoch_losses = np.concatenate([epoch_losses,np.array([epoch_loss])])
            
        if verbose:
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
            
        del loss
            
        #tensorboard observers
        grid_rgb_x = plot_rgb(X.cpu().detach(),vis_params['IMAGE_SAMPLES'],finetune_loader_val.dataset.output_bands,vis_params['RGB_BANDS'],channel_stats,'S2',True)
        grid_rgbsyn_x = plot_rgb(X.cpu().detach(),vis_params['IMAGE_SAMPLES'],finetune_loader_val.dataset.bands,vis_params['SYNTH_RGB'],channel_stats,'S1',True)
        grid_y_cat = plot_categorical(Y.cpu().detach(),vis_params['IMAGE_SAMPLES'],finetune_loader_val.dataset.legend_palette)
        grid_yhat_cat = plot_categorical(Y_hat.cpu().detach(),vis_params['IMAGE_SAMPLES'],finetune_loader_val.dataset.legend_palette)
        writer.add_images('finetune_input',grid_rgb_x,epoch)
        writer.add_images('finetune_input_synth',grid_rgbsyn_x,epoch)
        writer.add_images('finetune_target_categorical',grid_y_cat,epoch)
        writer.add_images('finetune_output_categorical',grid_yhat_cat,epoch)
            
        
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            
        #print('epoch losses: ',epoch_losses)
        if epoch>=finetune_params['EPOCH_BREAK_WINDOW']+1:
            epoch_losses_ave = np.convolve(epoch_losses - np.roll(epoch_losses,1), np.ones(finetune_params['EPOCH_BREAK_WINDOW']), 'valid') / finetune_params['EPOCH_BREAK_WINDOW']
            #print (epoch_losses_ave)
            if epoch_losses_ave[-1]>0:
                print ('val loss increasing -> break.')
                break

    # save model after training
    # save model after training
    if torch.cuda.device_count() > 1:
        torch.save(finetune_model.module.state_dict(), os.path.join(os.getcwd(), 'tmp','finetune_model.pth'))
    else:
        torch.save(finetune_model.state_dict(), os.path.join(os.getcwd(), 'tmp','finetune_model.pth'))
    #torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), 'tmp','finetune_optimizer.pth'))
