import os, json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsentinel.exp import ex
from deepsentinel.models.visualisation import plot_rgb, plot_categorical


def loss_fn(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp()) # splodes the loss so muchly??
    return MSE + KLD
    

@ex.capture
def pretrain_vae(model, 
             pretrain_loader, 
             optimizer, 
             writer,
             pretrain_params,  
             channel_stats,
             vis_params,
             device, 
             verbose=False):
    
    
    n_iter = 0
    if not verbose:
        epoch_pbar = tqdm(total = pretrain_params['EPOCHS'])
        epoch_pbar.set_description(f'Epochs')
    
    if channel_stats:
        channel_stats = json.load(open(channel_stats,'r'))
    
    epoch_losses = np.array([])
    for epoch in range(1, pretrain_params['EPOCHS'] + 1):
        
        cum_loss = 0
        if verbose:
            epoch_pbar = tqdm(total = len(pretrain_loader.dataset))
            epoch_pbar.set_description(f'Epoch {epoch}')
        
        
        for batch_idx, (X, Y) in enumerate(pretrain_loader):
            n_iter +=1
            model.train()
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            Y_hat, mu, logvar = model(X)

            
            loss = F.mse_loss(Y_hat, Y)
            #loss = loss_fn(Y_hat, Y, mu, logvar)
            loss.backward()
            optimizer.step()
            cum_loss+=loss.detach().item()

            if (batch_idx % pretrain_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                if verbose:
                    n_sample = (batch_idx * pretrain_params['BATCH_SIZE'])

                    
                    desc = f'Epoch {epoch} - avgloss {cum_loss/n_sample:.6f}'
                    epoch_pbar.set_description(desc)
            epoch_pbar.update(X.shape[0])

                
        epoch_loss = cum_loss/len(pretrain_loader.dataset)
        epoch_losses = np.concatenate([epoch_losses,np.array([epoch_loss])])
                
        writer.add_scalar('Pretrain_vae_Loss/train', epoch_loss, n_iter)
        del loss
            
        #tensorboard observers
        grid_rgb_x = plot_rgb(X.cpu().detach(),vis_params['IMAGE_SAMPLES'],pretrain_loader.dataset.bands,vis_params['RGB_BANDS'],channel_stats,'S2',True)
        grid_rgb_yh = plot_rgb(Y_hat.cpu().detach(),vis_params['IMAGE_SAMPLES'],pretrain_loader.dataset.bands,vis_params['RGB_BANDS'],channel_stats,'S2',True)
        writer.add_images('pretrain_input_rgb',grid_rgb_x,epoch)
        writer.add_images('pretrain_output_rgb',grid_rgb_yh,epoch)
            
        if verbose:
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            
        print('epoch losses: ',epoch_losses)
        if epoch>=pretrain_params['EPOCH_BREAK_WINDOW']:
            epoch_losses_ave = np.convolve(epoch_losses - np.roll(epoch_losses,1), np.ones(pretrain_params['EPOCH_BREAK_WINDOW']), 'valid') / pretrain_params['EPOCH_BREAK_WINDOW']
            print (epoch_losses_ave)
            if np.abs(epoch_losses_ave[-1])<pretrain_params['LOSS_CONVERGENCE']:
                print ('loss converged -> break.')
                break
        
        
    # save model after training
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_model.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_model.pth'))
    #torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_optimizer.pth'))
