import os, json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepsentinel.exp import ex
from deepsentinel.models.visualisation import plot_rgb, plot_categorical


def triplet_loss(z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd


@ex.capture
def pretrain_t2v(model, 
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
        
        
        for batch_idx, (a,n,d) in enumerate(pretrain_loader):
            n_iter +=1
            model.train()
            a, n, d = a.to(device), n.to(device), d.to(device)
            e_a = model(a)
            e_n = model(n)
            e_d = model(d)
            
            optimizer.zero_grad()
            loss, l_n, l_d, l_nd = triplet_loss(e_a, e_n, e_d)
            loss.backward()
            optimizer.step()
            cum_loss+=loss.detach().item()
            
            #sum_loss += loss.data[0]
            #sum_l_n += l_n.data[0]
            #sum_l_d += l_d.data[0]
            #sum_l_nd += l_nd.data[0]
            #if (idx + 1) * dataloader.batch_size % print_every == 0:
            #    print_avg_loss = (sum_loss - print_sum_loss) / (
            #        print_every / dataloader.batch_size)
            #    print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
            #        epoch, (idx + 1) * dataloader.batch_size, n_train,
            #        100 * (idx + 1) / n_batches, print_avg_loss))
            #    print_sum_loss = sum_loss
            
            #optimizer.zero_grad()
            #Y_hat, mu, logvar = model(X)
            #print ('Y_hat',Y_hat.shape)
            #print ('Y)',Y.shape)
            #print ('mu',mu.shape)
            #print ('logvar',logvar.shape)
            #loss = F.mse_loss(Y_hat, Y)
            #loss = loss_fn(Y_hat, Y, mu, logvar)
            #loss.backward()
            #optimizer.step()

            if (batch_idx % pretrain_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                if verbose:
                    n_sample = (batch_idx * pretrain_params['BATCH_SIZE'])

                    
                    desc = f'Epoch {epoch} - avgloss {cum_loss/n_sample:.6f}'
                    epoch_pbar.set_description(desc)
            epoch_pbar.update(a.shape[0])
                    
                    
        epoch_loss = cum_loss/len(pretrain_loader.dataset)
        epoch_losses = np.concatenate([epoch_losses,np.array([epoch_loss])])
                
                
        writer.add_scalar('Pretrain_t2v_Loss/train', loss.detach().item(), epoch)
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
    torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_optimizer.pth'))
