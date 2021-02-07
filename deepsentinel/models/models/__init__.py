from deepsentinel.models.models.vae import VAE
from deepsentinel.models.models.synthetic_rgb import SyntheticRGB
from deepsentinel.models.models.simple_fcnn import SimpleFCNN, SimpleCNN
from deepsentinel.models.models.mining_model import MinesClassifier
from deepsentinel.models.models.tilenet import TileNet
from deepsentinel.models.models.aegan import build_aegan

models = {
    'VAE':VAE,
    'synthetic_rgb':SimpleFCNN,#SyntheticRGB,
    'landcover':SimpleFCNN,
    'mining':MinesClassifier,
    'tile2vec':TileNet,
    'contrastive_loss':SimpleCNN,
    'aegan':build_aegan,
}
