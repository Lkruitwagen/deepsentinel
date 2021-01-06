

from deepsentinel.exp import ex


def grad_step(model, optimizer, X, Y, model_params):
    

@ex.capture
def pretrain(model, 
             pretrain_loader, 
             optimizer, 
             writer,
             pretrain_params, 
             model_params, 
             device, 
             verbose=False):
    
    
    
    
    
    n_iter = 0
    for epoch in range(1, pretrain_params['EPOCHS'] + 1):
        for batch_idx, (X, Y) in enumerate(pretrain_loader):
            model.train()
            X, Y = X.to(device), Y.to(device)
            loss = grad_step(model, optimizer, X, Y, model_params)
            n_iter += 1
            if batch_idx % train_params['LOG_INTERVAL'] == 0:
                if verbose:
                    n_sample = (batch_idx * train_params['BATCH_SIZE']) + len(X)
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, n_sample,
                                                                                   len(train_loader.dataset),
                                                                                   100. * n_sample / len(
                                                                                       train_loader.dataset),
                                                                                   loss.detach().item()))

                writer.add_scalar('Loss/train', loss.detach().item(), n_iter)
                writer.add_scalar('Loss/s_ent_loss', s_ent_loss.detach().item(), n_iter)
                writer.add_scalar('Loss/b_ent_loss', b_ent_loss.detach().item(), n_iter)

            del loss
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