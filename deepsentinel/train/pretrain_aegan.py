import os, json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsentinel.exp import ex
from deepsentinel.models.visualisation import plot_rgb, plot_categorical


@ex.capture
def pretrain_aegan(models, 
             pretrain_loader, 
             optimizers, 
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
        
    criterion = nn.BCELoss() # Dz
    criterion_l1 = nn.L1Loss() # x, x^
    criterion_l2 = nn.MSELoss() # z, z^
    criterion_NLL = nn.NLLLoss2d() # Dx 
    
    ## initialise weights
    # TODO
    
    epoch_losses = np.array([])
    for epoch in range(1, pretrain_params['EPOCHS'] + 1):
        
        cum_loss = 0
        if verbose:
            epoch_pbar = tqdm(total = len(pretrain_loader.dataset))
            epoch_pbar.set_description(f'Epoch {epoch}')
        
        
        for batch_idx, (X, _) in enumerate(pretrain_loader):
            
            for kk in models.keys():
                models[kk].train()
            
            X = X.to(device)
            
            # -------------------------
            # get labels and embeddings
            # -------------------------
            
            Z = noise_generator()                                       # random noise in latent
            Y_pos = torch.FloatTensor(X.shape[0]).fill_(1.)             # real image sample labels
            Y_neg = torch.FloatTensor(X.shape[0]).fill_(0.)             # fake latent sample labels
            Z_hat = models['encoder'](X)                                # real latent embeddings
            X_hat = models['decoder'](Z)                                # fake images generated from noise
            X_tilde = models['decoder'](Z_hat)                          # reconstructed real images
            Z_tilde = models['encoder'](X_hat)                          # reconstructed noise from fake images
            Dz_pos = models['discriminator_latent'](Z_hat)              # latent discriminator predictions on real embeddings
            Dz_neg = models['discriminator_latent'](Z)                  # latent discriminator predictions on fake noise
            Dx_pos = models['discriminator_image'](X)                   # image distriminator predictions on real images
            Dx_neg = models['discriminator_image'](X_hat)               # image discriminator prediction on fake images
            
            # ---------------
            # Train encoder and decoder as AE
            # ---------------

            X_reconstr_loss = criterion_l1(X_tilde, X)
            Z_reconstr_loss = criterion_l2(Z_tilde, Z)
            #X_reconstr_loss.backward()
            #Z_reconstr_loss.backward()
            reconstr_loss = X_reconstr_loss + Z_reconstr_loss
            reconstr_loss.backward()
            optimizers['encoder'].step()
            optimizers['decoder'].step()
            models['decoder'].zero_grad()
            models['encoder'].zero_grad()
            
            # ----------------
            # Train discriminators as GAN
            # AND update encoder/decoder -> perhaps this can't be simultaneous?
            # ----------------

            Dx_pos_loss = criterion(Dx_pos, Y_pos)
            Dx_neg_loss = criterion(Dx_neg, Y_neg)
            Dx_loss = Dx_poss_loss + Dx_neg_loss
            Dx_loss.backward()
            optimizers['discriminator_image'].step()
            optimizers['decoder-GAN'].step()
            
            Dz_pos_loss = criterion(Dz_pos, Y_pos)
            Dz_neg_loss = criterion(Dz_neg, Y_neg)
            Dz_loss = Dz_poss_loss + Dz_neg_loss
            Dz_loss.backward()
            optimizers['discriminator_latent'].step()
            optimizers['encoder-GAN'].step()
            
            for kk in optimizers.keys():
                optimizers[kk].zero_grad()
            
                    
        epoch_loss = cum_loss/len(pretrain_loader.dataset)
        epoch_losses = np.concatenate([epoch_losses,np.array([epoch_loss])])
                
                
        writer.add_scalar('Pretrain_csf_Loss/train', loss.detach().item(), epoch)
        del loss
            
        #tensorboard observers
            
        if verbose:
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
        del a, n, d
        del e_a, e_n, e_d
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            
        print('epoch losses: ',epoch_losses)
        if epoch>=pretrain_params['EPOCH_BREAK_WINDOW']:
            epoch_losses_ave = np.convolve(epoch_losses - np.roll(epoch_losses,1), np.ones(pretrain_params['EPOCH_BREAK_WINDOW']), 'valid') / pretrain_params['EPOCH_BREAK_WINDOW']
            print (epoch_losses_ave)
            if epoch_losses_ave[-1]<pretrain_params['LOSS_CONVERGENCE']:
                print ('loss converged -> break.')
                break

    # save model after training
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_model.pth'))
    else:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_model.pth'))
    #torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_optimizer.pth'))
