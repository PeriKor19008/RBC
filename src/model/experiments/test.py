from __future__ import annotations

from sympy.physics.units import frequency
from src.model.noise import *
from src.utils.paths import rel_to_root
from tests_helper import *
from occlusion import *
from frequency import *
from GradCAM import *

LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]

def test_avg_error(model: nn.Module, dir_path: str | Path, save_path_pct: str | None = None, thresh: float = 15.0,
                   block:bool = False,jitter:bool = False,noise:bool = False):
    dir_path = Path(dir_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    model.eval()
    dev = next(model.parameters()).device

    # accumulators (sum across samples)
    error = torch.zeros(4)
    error_prc = torch.zeros(4)
    max_prc_err = torch.zeros(4)
    it = 0

    for f in dir_path.iterdir():
        if not f.is_file() or f.suffix.lower() != ".f06":
            continue
        img, lbl_true = load_rbc_txt_image_and_labels(f)
        if block:
            img = change_block(1,img)
        if jitter:
            img = jitter_block(7,img,5)

        if noise:
            n = nn.Sequential(
                AddGaussianNoise(std=0.0, p=0.5),
                AddSpeckleNoise(std=0.6, p=0.5),
            )
            img = n(img)
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
            max_prc_err = [max(max_prc_err[i], prc_err[i]) for i in range(4)]
            it += 1
    # averages per label
    avg_abs_err = [error[i] / it for i in range(len(error))]
    avg_prc_err = [error_prc[i] / it for i in range(len(error_prc))]

    plot_error_prc(it, avg_prc_err, max_prc_err, str(save_path_pct))
    print("######")
    avg_err = 0
    for i in range(len(avg_prc_err)):
        avg_err += avg_prc_err[i]
    avg_err /= len(avg_prc_err)
    print("avg error------" + str(avg_err))
    print(" avg per label error----" + str(avg_prc_err))

def cor_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = rel_to_root(
        "outputs/models/FlexibleCNN/noise_BEST_old6_20251105-112939_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004819.pt")
    data_dir = rel_to_root("Data/extra_runs_for_check")
    data_dir_good = rel_to_root("Data/res_to_test")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module

    test_avg_error(model,data_dir_good,str(out_pct),99,block=False,jitter=False,noise=True)
    #test_erase(model,data_dir_good,str(out_pct))


def run_occlusion_demo(ckpt_path: str, sample_path: str,per_label: bool,avg:bool):
    # 1) load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(rel_to_root(ckpt_path),weights_only=False, map_location="cpu").to(device).eval()

    # 2) load one image + labels (no extra normalization here)
    sample_path = rel_to_root((sample_path))
    if not avg:
        img, lbl_true = load_rbc_txt_image_and_labels(sample_path)  # img: [1,50,50], lbl_true: [4]


    k = 5
    stride = 2
    if per_label and (not avg):
        heat, base_vec = occlusion_map_per_label(
            model=model,
            img=img,
            lbl_true=lbl_true,
            k=k,
            stride=stride,
            fill="mean",  # neutral occlusion
            eps=1e-8
        )

        print("Baseline per-label % error:",
              {k: float(v) for k, v in zip(LABEL_KEYS, base_vec.tolist())})


        plot_occlusion_maps_per_label(
            img=img,
            heat=heat,
            k=k,
            stride=stride,
            label_idx=None,  # None -> all labels
        )



    elif not per_label and not avg:
        heat, baseline = occlusion_map_simple(
            model=model,
            img=img,
            lbl_true=lbl_true,
            k=k,
            stride=stride,
            fill="mean",      # neutral occlusion
            eps=1e-8
        )
        print(f"Baseline macro % error (no occlusion): {baseline:.2f}%")

        # 4) plot + (optionally) save
        plot_occlusion_map_simple(
            img=img,
            heat=heat,
            k=k,
            stride=stride,
            title="Occlusion Δ% (macro)",

        )
    elif per_label and avg:
        avg_heat, avg_base, n, ref_img = occlusion_map_per_label_avg(
            model,
            dir_path=sample_path,
            k=k,
            stride=stride,
            fill="mean",  # neutral occlusion
            eps=1e-8,
        )




        plot_occlusion_maps_per_label(
            img=ref_img,
            heat=avg_heat,
            k=k,
            stride=stride,
            label_idx=None,  # None -> show all 4 label maps

        )
    else:
        avg_heat, avg_baseline, n, ref_img = occlusion_map_simple_avg(
            model,
            dir_path=sample_path,
            k=k,
            stride=stride,
            fill="mean",
            eps=1e-8,
        )
        print(f"Images averaged: {n} | Avg baseline macro %% error: {avg_baseline:.2f}%")

        # 4) plot + save
        out_png = rel_to_root("outputs/occlusion/avg_occlusion_macro.png")
        plot_occlusion_map_simple(
            img=ref_img,
            heat=avg_heat,
            k=k,
            stride=stride,
            title="Average Occlusion Δ% (macro)",
        )

def frequency_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = rel_to_root(
        "outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module
    data_dir = rel_to_root("Data/extra_runs_good_img")

    #test_gaussian_blur_sweep(model,data_dir)
    test_unsharp_sweep(model,data_dir)

def run_grad_Cam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = rel_to_root(
        "outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module
    img_path = rel_to_root("Data/res_to_test/06_097761827746a.f06")
    img, lbl = load_rbc_txt_image_and_labels(img_path)  # img: [1,50,50], lbl_true: [4]
    img = img.unsqueeze(0).to(device)
    gradcam = GradCAM(model)
    k=3
    cam = gradcam(img,target_index=k)

    plot_grad_cam(cam, img, k)





if __name__ == "__main__":
    #print (a_infer_ref_index_from_path(Path("../../../Data/extra_runs_for_check/20_0737523754741a.f06")))

    #print(load_rbc_txt_image_and_labels("../../../Data/extra_runs_for_check/05_0362516257251a.f06"))

    run_occlusion_demo(
        ckpt_path="outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt",
        sample_path="Data/extra_runs_good_img",per_label=False,avg=True,
    )

    #cor_run()
    #run_grad_Cam()
    #frequency_test()






