from src.model.ae_heads import AERegressor
from src.model.model import FCAutoencoder
from src.model.training.loops import train_model_val_loss
from torch.utils.data import DataLoader, random_split
from src.model.RBCDataset import RBCDatasetDB
from Data.DB_setup.db_config import DB_CONFIG
import torch, os


def train_ae_regressor_head(epochs,learning_rate,latent_dim,layers,model_ckpt: str = None):
# build dataloader
    full_ds = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)  # your dataset that returns (img, y[4])
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    BATCH = 32
    dls = {
        "train": DataLoader(train_ds, batch_size=BATCH, shuffle=True),
        "val":   DataLoader(val_ds,   batch_size=BATCH, shuffle=False),
    }



#create regressor from trained AE
    AE_CKPT = "../../"+ model_ckpt
    reg = AERegressor.from_checkpoint(
        ae_builder=FCAutoencoder,
        ckpt_path=AE_CKPT,
        ae_kwargs={"latent_dim": latent_dim, "hidden_dims": [1024, 512, 128]},  # use the dims you trained with check .json of the model
        latent_dim=latent_dim,                         # same latent as AE
        head_hidden=layers,                 # tiny MLP head
        freeze_encoder=True,                   # freeze AE encoder
        dropout=0.1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )



#train regressor
    criterion = torch.nn.MSELoss()            # or torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(reg.parameters(), lr=learning_rate)

    train_losses, val_losses, run_dir = train_model_val_loss(
        model=reg,
        dataloaders=dls,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        batch_size=BATCH,
        learning_rate=learning_rate,
        layers="AE64_head128x64_frozen",      # will appear in your logs/figs
        # if you wired schedulers.py, you can add:
        scheduler_name="onecycle",
        scheduler_params={
            "max_lr": 5e-3, "pct_start": 0.3, "div_factor": 5.0,
            "final_div_factor": 1e4, "cycle_momentum": False
        },
    )




#fine tune encoder
    reg.unfreeze_last_encoder_layers(n_linear=1)  # unlock last encoder layer

    opt = torch.optim.Adam(
        reg.param_groups(encoder_lr=1e-4, head_lr=1e-3, weight_decay=1e-5)
    )
    train_model_val_loss(
        model=reg,
        dataloaders=dls,
        criterion=criterion,
        optimizer=opt,
        num_epochs=10,
        batch_size=BATCH,
        learning_rate=1e-3,  # used only for logging tag in your loop
        layers="AE64_head128x64_ft-last1",
    )

# save model
    torch.save(reg, os.path.join(run_dir, "ae_regressor_full.pt"))

