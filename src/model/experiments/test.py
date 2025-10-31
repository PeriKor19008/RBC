from __future__ import annotations
from src.utils.paths import rel_to_root
from tests_helper import *




LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]

def test_avg_error(model: nn.Module, dir_path: str | Path, save_path_pct: str | None = None, thresh: float = 15.0,
                   block:bool = False,jitter:bool = False):
    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    model.eval()
    dev = next(model.parameters()).device

    # accumulators (sum across samples)
    error = torch.zeros(4)
    error_prc = torch.zeros(4)
    it = 0

    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue
        img, lbl_true = load_rbc_txt_image_and_labels(f)
        if block:
            img = change_block(1,img)
        if jitter:
            img = jitter_block(7,img,5)
        x = img.unsqueeze(0).to(dev)

        with torch.no_grad():
            lbl_pred = model(x).squeeze(0).detach().cpu()
        abs_err = abs(lbl_true - lbl_pred)
        eps = 1e-8
        prc_err = ((lbl_pred - lbl_true.cpu()).abs() / (lbl_true.cpu().abs() + eps) * 100.0).tolist()
        if any(v > thresh for v in prc_err):

            print(f"{f.name}: " + ",\t".join(f"{LABEL_KEYS[i]}={prc_err[i]:.2f}%" for i in range(4)))

            true_vals = [float(lbl_true[i].cpu()) for i in range(4)]
            print("\t" + "\t" + "\t".join(f"true_{LABEL_KEYS[i]}={true_vals[i]:.6g}" for i in range(4)))
        else:
            # element-wise accumulate
            error = [error[i] + abs_err[i] for i in range(len(abs_err))]
            error_prc = [error_prc[i] + prc_err[i] for i in range(len(prc_err))]
            it += 1
    # averages per label
    avg_abs_err = [error[i] / it for i in range(len(error))]
    avg_prc_err = [error_prc[i] / it for i in range(len(error_prc))]

    plot_avg_error_prc(it, avg_prc_err, str(save_path_pct))
    print("######")
    print(" avg abs error____" + str(avg_abs_err))
    print(" avg pct error----" + str(avg_prc_err))

def cor_multi_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = rel_to_root(
        "outputs/models/FlexibleCNN/3of_round3_BEST_20251028-201152_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.006.pt")
    data_dir = rel_to_root("Data/extra_runs_for_check")
    data_dir_good = rel_to_root("Data/extra_runs_good_img")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module

    test_avg_error(model,data_dir_good,str(out_pct),9900,block=True,jitter=False)
    #test_erase(model,data_dir_good,str(out_pct))

if __name__ == "__main__":
    #print (a_infer_ref_index_from_path(Path("../../../Data/extra_runs_for_check/20_0737523754741a.f06")))
    #multi_run()
    #single_run()
    #print(load_rbc_txt_image_and_labels("../../../Data/extra_runs_for_check/05_0362516257251a.f06"))


    cor_multi_run()



