from src.model.experiments.ae_reggresor import train_ae_regressor_head
from src.model.experiments.cnn import train_CNN, multi_train_CNN
from src.model.experiments.autoencoder import run_autoencoder, multi_train_autoencoder
from experiments.autoencoder import *


def train_single_model():
    #placeholders




    train_CNN(32, 25, 0.005,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),("conv",240)], [128,250], True)

    train_CNN(32, 25, 0.0008,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
              [128, 250], True)

    train_CNN(32, 35, 0.005,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
              [128, 250], True)

    train_CNN(32, 35, 0.0008,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
              [128, 250], True)

    train_CNN(32, 45, 0.005,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
              [128, 250], True)

    train_CNN(32, 45, 0.0008,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120), ("conv", 240)],
              [128, 250], True)

    #_, _, rd = run_autoencoder(32, 50, 0.0001, [1024, 512, 128], 64,True)


if __name__ == "__main__":
    train_single_model()
    #multi_train_CNN()
    #train_ae_regressor_head()

