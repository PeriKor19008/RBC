from src.model.experiments.cnn import train_CNN
from src.utils.paths import rel_to_root
import torch


def train_single_model():
    #placeholders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_path = rel_to_root(
        "outputs/models/FCAutoencoder/cor_noise_20251123-193317_FCAutoencoder_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt")
    ae_model = torch.load(ae_path, map_location="cpu", weights_only=False).to(device).eval()

    train_CNN(32, 25, 0.001,
              [("conv", 4), ("conv", 8), ("conv", 16), ("conv", 32), ("conv", 64), ("conv", 120),("conv",240)],
              [128,250],
              False,ae_model)

    #_, _, rd = run_autoencoder(32, 1, 0.001, [2048, 1024, 512, 128], 64, True)


    #_, _, rd = run_autoencoder(32, 25, 0.001, [1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [2048,1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [4000,2048,1024, 512, 128], 64)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [1024, 512, 128, 64], 32)
    # _, _, rd = run_autoencoder(32, 25, 0.001, [2048,1024, 512, 128,64], 32)






if __name__ == "__main__":
    train_single_model()
    #multi_train_CNN()
    #train_ae_regressor_head(25,0.001,64,(128,64),"outputs/models/FCAutoencoder/20251117-162720_FCAutoencoder_e35_lr0.0008_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt")

