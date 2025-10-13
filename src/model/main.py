from src.model.experiments.ae_reggresor import train_ae_regressor_head
from src.model.experiments.cnn import train_CNN, multi_train_CNN
from src.model.experiments.autoencoder import run_autoencoder, multi_train_autoencoder
from experiments.autoencoder import *


def train_single_model():
    #placeholders
    #train_CNN(32,1, 0.001,[("conv", 16), ("conv", 32), ("conv", 64)], [128])
    _, _, rd = run_autoencoder(32, 50, 0.0001, [1024, 512, 128], 64)


if __name__ == "__main__":
    #train_single_model()
    #multi_train_CNN()
    train_ae_regressor_head()

