def plot_all_val_losses(all_val_losses, filename="all_val_losses.png"):
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))
    for label, val_loss in all_val_losses.items():
        plt.plot(range(1, len(val_loss) + 1), val_loss, label=label)

    plt.title("Validation Loss Across CNN Configurations")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True)

    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", filename)
    plt.savefig(path)
    plt.close()
    print(f"All validation losses plot saved to: {path}")