from __future__ import annotations

from sympy.physics.units import frequency
from src.model.noise import *
from src.utils.paths import rel_to_root
from src.utils.show_image import display_image
from tests_helper import *
from occlusion import *
from frequency import *
from GradCAM import *
from src.model.experiments.ae_test_helpers import test_ae, load_ae_model
LABEL_KEYS = ["diameter", "thickness", "ratio", "ref_index"]





def cor_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = rel_to_root(
        "outputs/models/AERegressor/20251114-160908_AERegressor_e15_lr0.001_bs32_wd0.0_seed42_dsmanual/ae_regressor_full.pt")
    data_dir = rel_to_root("Data/extra_runs_for_check")
    data_dir_good = rel_to_root("Data/res_to_test")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module

    test_avg_error(model,data_dir_good,str(out_pct),25,block=False,jitter=False,noise=True)

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
        "outputs/models/FlexibleCNN/noise_BEST_old6_20251105-112939_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004819.pt")
    model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device).eval()  # <— full module
    data_dir = rel_to_root("Data/extra_runs_good_img")

    test_gaussian_blur_sweep(model,data_dir)
    #test_unsharp_sweep(model,data_dir)

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

def run_ae_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "outputs/models/FCAutoencoder/20251113-053246_FCAutoencoder_e45_lr0.0008_bs32_wd0.0_seed42_dsmanual/autoencoder_final.pt"
    data_dir_good = rel_to_root("Data/res_to_test")
    out_pct = rel_to_root("outputs/test_graphs/extra_runs_avg_pct_error.png")
    model = load_ae_model(ckpt_path,64,[1024,512,128])

    test_ae(model, data_dir_good, str(out_pct))




if __name__ == "__main__":
    #print (a_infer_ref_index_from_path(Path("../../../Data/extra_runs_for_check/20_0737523754741a.f06")))

    #print(load_rbc_txt_image_and_labels("../../../Data/extra_runs_for_check/05_0362516257251a.f06"))

    # run_occlusion_demo(
    #     ckpt_path="outputs/models/FlexibleCNN/BEST_old6_20251104-070605_FlexibleCNN_e25_lr0.001_bs32_wd0.0_seed42_dsmanual/FlexibleCNN_e25_lr0.001_bs32_val0.004462.pt",
    #     sample_path="Data/extra_runs_good_img",per_label=False,avg=True,
    # )

    cor_run()
    #run_grad_Cam()
    #frequency_test()
    #run_ae_test()






