import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from deepsentinel.exp import ex
    

@ex.capture
def pretrain_loop(model, 
             pretrain_loader, 
             optimizer, 
             writer,
             pretrain_params,  
             device, 
             verbose=False):
    
    
    n_iter = 0
    if not verbose:
        epoch_pbar = tqdm(total = pretrain_params['EPOCHS'])
        epoch_pbar.set_description(f'Epochs')
    
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
            Y_hat = model(X)
            loss = F.mse_loss(Y_hat, Y)
            loss.backward()
            optimizer.step()

            if (batch_idx % pretrain_params['LOG_INTERVAL'] == 0) & (batch_idx >0):
                if verbose:
                    n_sample = (batch_idx * pretrain_params['BATCH_SIZE'])
                    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, n_sample,
                    #                                                               len(pretrain_loader.dataset),
                    #                                                               100. * n_sample / len(
                    #                                                                   pretrain_loader.dataset),
                    #                                                               loss.detach().item()))
                    cum_loss+=loss.detach().item()
                    desc = f'Epoch {epoch} - avgloss {cum_loss/n_sample:.6f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(X.shape[0]*pretrain_params['LOG_INTERVAL'])

                writer.add_scalar('Loss/train', loss.detach().item(), n_iter)

            del loss
            
        if verbose:
            epoch_pbar.close()
        else:
            epoch_pbar.update(1)
        del X
        del Y
        
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # save model after training
    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(os.getcwd(), 'tmp','pretrain_optimizer.pth'))


def train():
    pass

def test():
    pass