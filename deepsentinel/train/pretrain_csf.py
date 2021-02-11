import os, json
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

from deepsentinel.exp import ex
from deepsentinel.models.visualisation import plot_rgb, plot_categorical


@ex.capture
def pretrain_csf(model, 
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
        
    # intialise the parameters on the dataloader
    pretrain_loader.dataset._epoch_end(0, pretrain_params['EPOCHS'])
    
    epoch_losses = np.array([])
    for epoch in range(1, pretrain_params['EPOCHS'] + 1):
        
        cum_loss = 0
        if verbose:
            epoch_pbar = tqdm(total = len(pretrain_loader.dataset))
            epoch_pbar.set_description(f'Epoch {epoch}')
        
        
        for batch_idx, (v1, v2) in enumerate(pretrain_loader):
            n_iter +=1
            model.train()
            v1,v2 = v1.to(device), v2.to(device)
            e_v1 = model(v1)
            e_v2 = model(v2)
            #e_v1_stack = torch.transpose(torch.stack([e_v1]*(e_v2.shape[0]-1)),0,1)
            #e_v2_stack = torch.stack([e_v2[tuple(jj for jj in range(e_v2.shape[0]) if jj!=ii),:,:,:] for ii in range(e_v2.shape[0])])
            # print ('means',v1.mean(),v2.mean(),e_v1.mean(),e_v2.mean())
            
            optimizer.zero_grad()
            
            # batch cross tripletloss
            l_n = torch.sqrt((e_v1 - e_v2) ** 2+1e-8).sum(dim=1)
            l_d = - torch.sqrt((e_v1 - torch.roll(e_v2,int(np.random.choice(range(1,pretrain_params['BATCH_SIZE']-2))),dims=0)) ** 2 + 1e-8).sum(dim=1)
                               
            #loss = F.relu(l_n.mean()) + F.relu(l_d.mean())
            #loss = l_n+l_d
            loss = F.relu(l_n+l_d + 0.1) # margin-> 0.1??
            loss = torch.mean(loss)
            #loss = torch.mean(loss)
                               
            loss.backward()
            optimizer.step()
            cum_loss+=loss.detach().item()

            n_sample = ((batch_idx+1) * pretrain_params['BATCH_SIZE'])


            desc = f'Epoch {epoch} - avgloss {cum_loss/n_sample:.6f}'
            epoch_pbar.set_description(desc)
            epoch_pbar.update(v1.shape[0])
                    
                    
        epoch_loss = cum_loss/len(pretrain_loader.dataset)
        epoch_losses = np.concatenate([epoch_losses,np.array([epoch_loss])])
                
                
        writer.add_scalar('Pretrain_csf_Loss/train', loss.detach().item(), epoch)
        del loss
            
        #tensorboard observers
            
        if verbose:
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
        del v1, v2
        del e_v1, e_v2
        
        # update learning curriculum
        pretrain_loader.dataset._epoch_end(epoch, pretrain_params['EPOCHS'])
        
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
