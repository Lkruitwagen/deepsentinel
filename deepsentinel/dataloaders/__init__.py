from deepsentinel.dataloaders.vae import VAEDataloader
from deepsentinel.dataloaders.synthrgb import SynthRGBDataloader
from deepsentinel.dataloaders.landcover import CorineLandCover
from deepsentinel.dataloaders.mines import MinesLoader
from deepsentinel.dataloaders.tile2vec import Tile2VecLoader
from deepsentinel.dataloaders.contrastive_loss import ContrastiveLoss

dataloaders = {
    'VAE':VAEDataloader,
    'synthetic_rgb':SynthRGBDataloader,
    'landcover':CorineLandCover,
    'mines':MinesLoader,
    'tile2vec':Tile2VecLoader,
    'contrastive_loss':ContrastiveLoss,
}