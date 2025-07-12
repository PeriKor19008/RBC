import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import SimpleModel, CNNModel, FlexibleCNN  # Assuming you have a separate model.py
from RBCDataset import RBCDatasetDB  # Assuming you have your dataset class
from src.model.plot import plot_all_val_losses
from src.model.train import train_model_val_loss, get_next_run_number
from train import train_model  # function to train and log results
import matplotlib.pyplot as plt
import os
from train import plot_and_save_loss_graph, plot_loss_graphs
from Data.DB_setup.db_config import DB_CONFIG

#

def default_run_NN(lr,bs,l,ne):
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



def train_CNN(batchSize, epochs, layers):
    # ---create datasets---
    full_dataset = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # -----create dataloader----
    batch_size = batchSize
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    }

    #---Init model, loss, opt---
    all_val_losses = {}  # Dictionary to collect val losses with a label

    conv_configs = [
        [("conv", 16), ("conv", 32), ("conv", 64)],
        [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 64)],
        [("conv", 16), ("conv", 32), ("conv", 64), ("conv", 64), ("conv", 64)]
    ]
    fc_configs = [
        [128],
        [128, 256]
    ]

    for  c in conv_configs:
        for  f in fc_configs:
            model = FlexibleCNN(c, f)
            criterion = torch.nn.MSELoss()
            learning_rate = 0.001
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            #---define params---
            num_epochs = epochs
            layers = f"conv{len(c)}_fc{len(f)}"


            #--- train---
            train_losses, val_losses = train_model_val_loss(
                model=model,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                layers=layers
            )
            run_number = get_next_run_number()
            plot_loss_graphs(train_losses, val_losses,run_number, num_epochs, learning_rate, batch_size, layers)

            #--- save all val loss curves---
            all_val_losses[layers] = val_losses
    plot_all_val_losses(all_val_losses)

if __name__ == "__main__":
    os.makedirs("comp_graphs", exist_ok=True)
    train_CNN(32,30,"mult_run")






