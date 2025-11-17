from src.model.experiments.ae_reggresor import train_ae_regressor_head
from src.model.experiments.cnn import train_CNN, multi_train_CNN
from src.model.experiments.autoencoder import run_autoencoder, multi_train_autoencoder
from experiments.autoencoder import *


def train_single_model():
    #placeholders




    # train_CNN(32, 25, 0.001,
    #           [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),("conv",240)],
    #           [128,250],
    #           False)

    #_, _, rd = run_autoencoder(32, 1, 0.001, [2048, 1024, 512, 128], 64, True)

    #_, _, rd = run_autoencoder(32, 35, 0.001, [1024, 512, 128], 64,True)
    # _, _, rd = run_autoencoder(32, 45, 0.001, [1024, 512, 128], 64, True)
    # _, _, rd = run_autoencoder(32, 25, 0.0008, [1024, 512, 128], 64, True)
     _, _, rd = run_autoencoder(32, 35, 0.0008, [1024, 512, 128], 64, False)
    # _, _, rd = run_autoencoder(32, 45, 0.0008, [1024, 512, 128], 64, True)





if __name__ == "__main__":
    #train_single_model()
    #multi_train_CNN()
    train_ae_regressor_head(25,0.001,64,(128,64),"outputs/models/FCAutoencoder/20251117-162720_FCAutoencoder_e35_lr0.0008_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt")

