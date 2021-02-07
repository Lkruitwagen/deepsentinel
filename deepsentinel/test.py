import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchsummary import summary


from deepsentinel.models.models import models
from deepsentinel.models.encoders import encoders
from deepsentinel.train import pretrain_vae, finetune_synthrgb, finetune_lc, pretrain_t2v
from deepsentinel.dataloaders import dataloaders

"""A script to instantiate the experiment object"""
import os, glob, yaml

def mIOU(label, pred):

    # pred: BCWH
    N_classes = pred.shape[1]
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1).detach().numpy()
    label = label.view(-1).detach.numpy()
    recs = {}
    for cc in range(N_classes):
        A = (pred==cc)
        B = (label==cc)
        N = B.sum()
        I = A & B
        U = A | B
        if U.sum()==0:
            IoU=np.nan
        else:
            IoU=I.sum()/U.sum()
        
        recs[cc] = {'N':N,'IoU':IoU, 'I':I.sum(), 'U':U.sum()}
        
    mean_iou = np.mean([vv['IoU'] for vv,cc in recs.items()])
        
    return mean_iou, recs

def test(pretrain, finetune, load_run, encoder, encoder_params, encoder_layers, pretrain_config, pretrain_model_config, pretrain_loader_config,finetune_config, finetune_model_config, finetune_loader_config, vis_params, device, verbose, random_seed, trn_split, cv_split):
    """
    The test loop that loads each run and obtains final mean ious for the land classification problem.
    
    Returns
    -------
        None
    """
    
    test_results = {}
    
    
    for pretrain,dd in test_config.items():
        
        test_results[pretrain]={}
        
        for encoder, load_run in dd.items():
            

            finetune_dataset = dataloaders[finetune](**finetune_loader_config[finetune])

            finetune_loader_test = DataLoader(
                dataloaders[finetune](seed=random_seed, start_por=cv_split, end_por=1.,**finetune_loader_config[finetune]), 
                batch_size=finetune_config[finetune]['BATCH_SIZE'], 
                shuffle=False,
                num_workers=finetune_config[finetune]['DATALOADER_WORKERS'], 
            )

            _, _Y = next(iter(finetune_loader_test))
            print ('n classes', _Y.shape[1])

            test_model = models[finetune](encoder, encoder_params[encoder], n_classes=_Y.shape[1], **finetune_model_config[finetune])

            logger.info(f'Loading saved run {load_run}')

            logger.info('Loading saved run')
            finetuned_dict = torch.load(os.path.join(os.getcwd(), 'experiments', 'sacred', str(load_run), 'finetuned_model.pth'))
            # filter the keys needed for the encoder
            finetuned_dict = {kk:vv for kk,vv in finetuned_dict.items() if 'encoder' in kk}
            # get the state_dict for the finetuning model
            model_dict = test_model.state_dict()
            # update the state dict
            model_dict.update(finetuned_dict)
            # load the updated model_dict
            test_model.load_state_dict(model_dict)

            test_model.eval()
            
            _, _Y = next(iter(finetune_loader_test))
            
            N_classes = _Y.shape
            
            epoch_pbar = tqdm(total = pretrain_params['EPOCHS'])
            epoch_pbar.set_description(f'Testing pretain: {pretrain}; encoder: {encoder}')
            
            batch_recs = []
            with torch.no_grad():
                for batch_idx, (X, Y) in enumerate(finetune_loader_test):
                    X, Y = X.to(device), Y.to(device)
                    Y_hat = model(X)
                    
                    mIOU, batch_rec = mIOU(label, pred)
                
                    batch_recs.append(batch_rec)
                    
                    
            all_ious = {cc:{'I':0,'U':0} for cc in range(N_classes)}
            for rec in batch_recs:
                for cc in range(N_classes):
                    all_ious[cc]['I']+=rec[cc]['I']
                    all_ious[cc]['U']+=rec[cc]['U']
                    
            test_results[pretrain][encoder]=[all_ious[cc]['I']/all_ious[cc]['U'] for cc in range(N_classes)]
            print (test_results[pretrain][encoder])
            json.dump(test_results, open(os.path.join(os.getcwd(),'test_results.json'),'w'))