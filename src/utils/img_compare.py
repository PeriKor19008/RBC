import numpy as np
from Data.DB_setup.image_db_utils import ImageDB
import matplotlib.pyplot as plt
import os



def compare_images(image_data1, image_data2, use_log):
    if not image_data1:
        print(" No image data found.")
        return
    if not image_data2:
        print(" No image data found.")
        return

    img1_txt = image_data1['log_image'] if use_log else image_data1['regular_image']
    img2_txt = image_data2['log_image'] if use_log else image_data2['regular_image']

    img1 =[float(line.replace('D', 'E')) for line in img1_txt.strip().splitlines()]
    img2= [float(line.replace('D', 'E')) for line in img2_txt.strip().splitlines()]

    if len(img1)!=2500 or len(img2) != 2500:
        raise ValueError("expected 2500 values")
    img1 = np.array(img1)
    img2 = np.array(img2)

    diff =(img1 - img2)
    # Create and show image
    # Display with diverging colormap
    plt.imshow(diff, cmap='seismic', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(label='Difference (img1 - img2)')
    plt.title("Signed Image Difference (Color)")
    plt.axis('off')
    plt.savefig("img_cmp.png", dpi=300, bbox_inches='tight')
    plt.show()



def file_comp(path1,path2):
    img1 = read_fortran_float_file(path1)
    img2 = read_fortran_float_file(path2)

    diff = (img1 - img2)


    # Create and show image
    # Display with diverging colormap
    plt.imshow(diff, cmap='seismic', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar(label='Difference (img1 - img2)')
    plt.title("Signed Image Difference (Color)")
    plt.axis('off')

    plt.show()



def read_fortran_float_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Replace 'D' with 'E' so Python can parse the float correctly
    float_values = [float(line.strip().replace('D', 'E')) for line in lines]

    if len(float_values) != 2500:
        raise ValueError(f"Expected 2500 values, got {len(float_values)}")

    return np.array(float_values).reshape((50, 50))


def pixel_variance_deviation(file):
    folder_path = '../../Data/results/'

    # Count only directories
    folder_count = sum(
        os.path.isdir(os.path.join(folder_path, item))
        for item in os.listdir(folder_path)
    )

    img_sequence = []
    c = 5
    for f in range(folder_count):
        tmp = (f * 5) + c
        str_num = "{:03}".format(tmp)
        print(str_num)
        img = read_fortran_float_file(folder_path+"Refindx1."+ str_num +"/" + file)
        img_sequence.append(img)
    stack = np.stack(img_sequence)

    std_map = np.std(stack, axis=0)

    # Visualize
    plt.imshow(std_map, cmap='hot')
    plt.colorbar(label="Pixel Std. Dev.")
    plt.title("Per-pixel Change Across Sequence")
    plt.axis('off')
    plt.savefig("std_map_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

# === Example usage ===
if __name__ == '__main__':
    db = ImageDB()
    #compare_images(db.search_image_by_all_dtr(7000,2250,600,45),db.search_image_by_all_dtr(7000,2250,600,100),0)
    #file_comp("../../Data/results/Refindx1.005/0450015006001a.f06", "../../Data/results/Refindx1.075/0450015006001a.f06")
    pixel_variance_deviation("0450015006001a.f06")

