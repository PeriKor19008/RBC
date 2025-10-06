import numpy as np
import matplotlib.pyplot as plt

# Load pixel values, converting Fortran-style "D" to "E" and scaling to 0–255
with open('../../documentation/runs_vs_refractive_index_at_he_ne/data.txt', 'r') as file:
    pixel_values = [
        float(line.strip().replace('D', 'E')) for line in file if line.strip()
    ]

# Normalize values to 0–255
pixel_array = np.array(pixel_values)
min_val, max_val = pixel_array.min(), pixel_array.max()
normalized_pixels = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Reshape into image (e.g. 50x50 if you have 2500 values)
image = normalized_pixels.reshape((50, 50))

# Display the image
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()

# Optionally save the image
# plt.imsave('output_image.png', image, cmap='gray')
