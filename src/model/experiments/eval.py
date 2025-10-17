# sanity_eval.py
import torch
from torch.utils.data import DataLoader, random_split
from src.model.RBCDataset import RBCDatasetDB
from Data.DB_setup.db_config import DB_CONFIG

# load the exact regressor you trained
reg = torch.load("<path-to>/ae_regressor_full.pt", map_location="cpu").eval()
reg.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

ds = RBCDatasetDB(db_config=DB_CONFIG, use_log_image=False)   # same as training!
train_sz = int(0.8*len(ds)); val_sz = len(ds) - train_sz
_, val_ds = random_split(ds, [train_sz, val_sz])
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

mae = torch.zeros(4); n = 0
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(reg.head[0].weight.device)  # any model param device
        pred = reg(x).cpu()
        mae += (pred - y).abs().sum(dim=0)
        n += y.size(0)
mae /= n
print("Val MAE [diam, thick, ratio, ref_index]:", mae.tolist())
