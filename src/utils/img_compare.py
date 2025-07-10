from PIL import Image
import numpy as np
from Data.DB_setup.image_db_utils import ImageDB

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

    diff =(img1 - img2) * 100
    diff_img = Image.fromarray(diff)
    diff_img.show()

# === Example usage ===
if __name__ == '__main__':
    db = ImageDB()
    compare_images(db.search_image_by_all_dtr(7000,2250,600,45),db.search_image_by_all_dtr(7000,2250,600,100),0)



