from deepsentinel.dataloaders.vae import VAEDataloader
from deepsentinel.dataloaders.synthrgb import SynthRGBDataloader

dataloaders = {
    'VAE':VAEDataloader,
    'synthetic_rgb':SynthRGBDataloader,
}