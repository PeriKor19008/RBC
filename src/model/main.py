import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleModel  # Assuming you have a separate model.py
from RBCDataset import RBCDatasetDB  # Assuming you have your dataset class
from train import train_model  # function to train and log results
import matplotlib.pyplot as plt
import os
from train import plot_and_save_loss_graph
from Data.DB_setup.db_config import DB_CONFIG

#

def default_run(lr,bs,l,ne):
    learning_rate = lr
    batch_size = bs
    layers = l
    num_epochs = ne

    dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=True)
    print(f"Number of samples in dataset: {len(dataset)}")

    dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}
    # Initialize model, loss, and optimizer
    model = SimpleModel(layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    epoch_losses = train_model(model, dataloaders, criterion, optimizer,
                               num_epochs, batch_size, learning_rate, layers)
    minloss = min(epoch_losses)
    # Save training loss graph to comp_graphs
    os.makedirs("comp_graphs", exist_ok=True)
    run_id = id
    plot_and_save_loss_graph(epoch_losses, run_id, num_epochs, learning_rate, batch_size, layers,
                             save_dir="comp_graphs")


if __name__ == "__main__":
    os.makedirs("comp_graphs", exist_ok=True)

    default_run(0.001,32,[2500,1000,128,4],40)





