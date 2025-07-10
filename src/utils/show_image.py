import numpy as np
import matplotlib.pyplot as plt
from fontTools.unicodedata import block

from Data.DB_setup.image_db_utils import ImageDB


def display_image(image_data, use_log=True):
    if not image_data:
        print(" No image data found.")
        return

    image_text = image_data['log_image'] if use_log else image_data['regular_image']
    values = [float(line.replace('D', 'E')) for line in image_text.strip().splitlines()]

    if len(values) != 2500:
        raise ValueError("Expected 2500 values")

    image_array = np.array(values).reshape((50, 50))

    #  Print filename and filepath
    print(f" Filename: {image_data['filename']}")
    print(f" Filepath: {image_data['filepath']}")
    title = f"RefIdx: {image_data['ref_index']}  D: {image_data['diameter']}  T: {image_data['thickness']}  R: {image_data['ratio']}"
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.figure()



def plot_image_as_line(image_data, use_log=False):
    if not image_data:
        print(" No image data found.")
        return

    image_text = image_data['log_image'] if use_log else image_data['regular_image']
    values = [float(line.replace('D', 'E')) for line in image_text.strip().splitlines()]

    if len(values) != 2500:
        raise ValueError("Expected 2500 values")

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(values)), values, label='Image data')
    plt.xlabel('Data Index (0–2499)')
    plt.ylabel('Value')
    plt.title('Image Data as Line Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.figure()


# === Example usage ===
if __name__ == '__main__':
    db = ImageDB()
    image = db.search_image_by_dtr(7000,2250,600)
    db.close()

    plot_image_as_line(image, use_log=True)
    display_image(image,use_log=True)
    plt.show()
