import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SimpleModel  # Assuming you have a separate model.py
from RBCDataset import RBCDataset  # Assuming you have your dataset class
from train import train_model  # function to train and log results
import matplotlib.pyplot as plt
import os
from train import plot_and_save_loss_graph


def get_user_input():
    print("🔧 Enter training configuration:")

    try:
        num_epochs = int(input("Number of epochs (e.g. 30): "))
    except ValueError:
        num_epochs = 30
        print("Invalid input. Defaulting to 30 epochs.")

    try:
        learning_rate = float(input("Learning rate (e.g. 0.001): "))
    except ValueError:
        learning_rate = 0.001
        print("Invalid input. Defaulting to 0.001.")

    try:
        batch_size = int(input("Batch size (e.g. 32): "))
    except ValueError:
        batch_size = 32
        print("Invalid input. Defaulting to 32.")

    print("Define layer sizes. Example: 4096 256 128 3")
    try:
        layers = list(map(int, input("Layer sizes: ").strip().split()))
    except ValueError:
        layers = [64 * 64, 128, 3]
        print("Invalid input. Defaulting to [4096, 128, 3].")
    id = input("Enter Run ID")

    return num_epochs, learning_rate, batch_size, layers,id


def change_num_epoch():


    # Define training parameters
    learning_rate = 0.001
    batch_size = 32
    layers = [64 * 64, 128, 3]  # Example layers for the model

    # Initialize dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = RBCDataset(csv_file="../../mock_data/rbc_mock_dataset/labels.csv",
                         image_dir="../../mock_data/rbc_mock_dataset/images",
                         transform=transform)

    dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}

    # Initialize the model
    model = SimpleModel(layers)

    # Define your optimizer and loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # List to store the results
    epoch_list = []
    min_loss_list = []

    # Train the model for different numbers of epochs (10, 20, 30, ...)
    for run_index, num_epochs in enumerate(range(10, 101, 10), start=1):
        print(f"Run #{run_index} with {num_epochs} epochs")

        # Create new model and optimizer for each run
        model = SimpleModel(layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and get list of epoch losses
        epoch_losses = train_model(model, dataloaders, criterion, optimizer, num_epochs, batch_size, learning_rate,
                                   layers)

        min_loss = min(epoch_losses)
        epoch_list.append(num_epochs)
        min_loss_list.append(min_loss)



    # Plot min loss vs epochs across all runs
    plt.figure()
    plt.plot(epoch_list, min_loss_list, marker='o', color='green')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Minimum Loss")
    plt.title("Epochs vs Minimum Loss")
    plt.grid(True)
    plt.show()
    plt.savefig("comp_graphs/loss_vs_epochs.png")
    plt.close()

    print("Training complete and graph saved.")

def change_layer():


        # Define the parameters
        num_epochs = 30
        learning_rate = 0.001
        batch_size = 32

        layers_template = [[64 * 64, 128, 3] , [64 * 64, 256,128, 3],[64 * 64, 512, 256,128, 3], [64 * 64, 1024,512, 256,128, 3]]

        # Initialize dataset and dataloaders
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        dataset = RBCDataset(csv_file="../../mock_data/rbc_mock_dataset/labels.csv",
                             image_dir="../../mock_data/rbc_mock_dataset/images",
                             transform=transform)
        dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}

        # Initialize lists to track min losses and corresponding layer configurations
        min_loss_list = []
        layer_configurations = []

        # Run training 5 times, each time adding a layer
        for run_number in range(0, 4):
            layers = layers_template[run_number]  # Add corresponding layer size

            print(f"Running training for layers: {layers}")

            # Initialize the model
            model = SimpleModel(layers)

            # Define the optimizer and loss function
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Train the model and log results
            epoch_losses = train_model(model, dataloaders, criterion, optimizer, num_epochs, batch_size,
                                                 learning_rate, layers)
            min_loss=min(epoch_losses)

            # Track the min loss and layer configuration
            min_loss_list.append(min_loss)
            layer_configurations.append(len(layers))  # Number of layers in the model



        # After all runs, generate the final graph comparing number of layers vs. min loss
        plt.figure(figsize=(8, 6))
        plt.plot(layer_configurations, min_loss_list, marker='o', color='b', label='Min Loss')
        plt.title('Min Loss vs. Number of Layers')
        plt.xlabel('Number of Layers')
        plt.ylabel('Min Loss')
        plt.grid(True)
        plt.legend()

        # Save the graph in the 'comp_graphs' folder
        os.makedirs('comp_graphs', exist_ok=True)
        plt.savefig('comp_graphs/layers_vs_min_loss.png')
        plt.close()

        print("Training complete for all runs. The graph has been saved.")

def change_LR():
    # Fixed parameters
    num_epochs = 30
    batch_size = 32
    layers = [64 * 64, 128, 3]

    # Learning rates to try
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = RBCDataset(csv_file="../../mock_data/rbc_mock_dataset/labels.csv",
                         image_dir="../../mock_data/rbc_mock_dataset/images",
                         transform=transform)

    dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}

    # Lists to store learning rates and corresponding min losses
    min_loss_list = []

    for i, lr in enumerate(learning_rates, start=1):
        print(f"\n🔁 Run #{i} — Learning Rate: {lr}")

        # Create model
        model = SimpleModel(layers)

        # Define optimizer and loss function
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train
        epoch_losses= train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs,
            batch_size,
            lr,
            layers
        )
        min_loss=min(epoch_losses)

        # Track results
        min_loss_list.append(min_loss)



    # Plot comparison graph
    os.makedirs("comp_graphs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, min_loss_list, marker='o', linestyle='-', color='blue', label='Min Loss')
    plt.title("Learning Rate vs Min Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Minimum Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("comp_graphs/learning_rate_vs_min_loss.png")
    plt.close()

    print("✅ All runs complete. Comparison graph saved to comp_graphs/learning_rate_vs_min_loss.png.")

def change_batch_size():
    # Fixed training parameters
    num_epochs = 30
    learning_rate = 0.001
    layers = [64 * 64, 128, 3]

    # Batch sizes to test
    batch_sizes = [8, 16, 32, 64, 128]

    # Dataset transformation
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load dataset once
    dataset = RBCDataset(csv_file="../../mock_data/rbc_mock_dataset/labels.csv",
                         image_dir="../../mock_data/rbc_mock_dataset/images",
                         transform=transform)

    # Lists to collect results
    min_loss_list = []

    for i, batch_size in enumerate(batch_sizes, start=1):
        print(f"\n🔁 Run #{i} — Batch Size: {batch_size}")

        # Prepare dataloader
        dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}

        # Initialize model
        model = SimpleModel(layers)

        # Loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        epoch_losses = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs,
            batch_size,
            learning_rate,
            layers
        )
        min_loss=min(epoch_losses)



        # Track min loss for this batch size
        min_loss_list.append(min_loss)

    # Create summary graph
    os.makedirs("comp_graphs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(batch_sizes, min_loss_list, marker='o', linestyle='-', color='green', label='Min Loss')
    plt.title("Batch Size vs Min Loss")
    plt.xlabel("Batch Size")
    plt.ylabel("Minimum Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("comp_graphs/batch_size_vs_min_loss.png")
    plt.close()

    print("✅ All batch size runs complete. Comparison saved to comp_graphs/batch_size_vs_min_loss.png.")

def manual_input():
    # Get parameters from user
    num_epochs, learning_rate, batch_size, layers,id = get_user_input()

    # Show the configuration
    print("\n📋 Using configuration:")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {batch_size}")
    print(f"Layers: {layers}")

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset = RBCDataset(csv_file="../../mock_data/rbc_mock_dataset/labels.csv",
                         image_dir="../../mock_data/rbc_mock_dataset/images",
                         transform=transform)

    dataloaders = {'train': DataLoader(dataset, batch_size=batch_size, shuffle=True)}

    # Initialize model, loss, and optimizer
    model = SimpleModel(layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    epoch_losses = train_model(model, dataloaders, criterion, optimizer,
                                         num_epochs, batch_size, learning_rate, layers)
    minloss=min(epoch_losses)
    # Save training loss graph to comp_graphs
    os.makedirs("comp_graphs", exist_ok=True)
    run_id = id
    plot_and_save_loss_graph(epoch_losses, run_id, num_epochs, learning_rate, batch_size, layers, save_dir="comp_graphs")



if __name__ == "__main__":
    os.makedirs("comp_graphs", exist_ok=True)




    preference = input("if you want to change num epoch enter 0\n "
          "if you want to change layers enter 1 \n "
          "if you want to change batch size enter 2\n "
          "if you want to change learning rate enter 3\n "
          "if you want to run all enter 4\n"
          "if you want to set manually enter 5\n ")
    if preference == '0':change_num_epoch()
    elif preference == '1':change_layer()
    elif preference == '2':change_batch_size()
    elif preference == '3':change_LR()
    elif (preference == '4'):
        change_num_epoch()
        change_layer()
        change_LR()
        change_batch_size()
    elif (preference == '5'): manual_input()

