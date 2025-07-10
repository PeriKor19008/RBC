import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import SimpleModel
from src.utils.fileName_to_params import file_name_to_params

#---- Load Model ----
layers = [2500,500,128,4]
model = SimpleModel(layers)

model_num = input("enter model number: ")
state_dict = torch.load('models/model_run'+model_num+'.pt')
model.load_state_dict(state_dict)
model.eval()

# ---  Load and Prepare Image from TXT ---
def load_txt_image(path):
    with open(path, 'r') as f:
        values = []
        for line in f:
            line = line.strip().replace('D', 'E').replace('d', 'E')
            values.append(float(line))
        if len(values) != 2500:
            raise ValueError(f"Expected 2500 pixel values, got {len(values)}")
    image = np.array(values, dtype=np.float32).reshape((50, 50))
    image = torch.from_numpy(image).unsqueeze(0)  # Shape: [1, 50, 50] — 1 channel
    return image.unsqueeze(0)

transform = transforms.Compose([
    transforms.Resize((50, 50)),          # Ensure size is correct
    transforms.ToTensor(),                # Converts to [C, H, W] format
    # transforms.Normalize(mean, std),    # Uncomment if your model was trained with normalization
])


# Apply transform
input_tensor = load_txt_image("../../Data/results/Refindx1.095/0450015007001b.f06")  # Add batch dimension: [1, C, 50, 50]

#---- Run Model
with torch.no_grad():  # Disable gradient tracking
    output = model(input_tensor)
    result = output.squeeze().tolist()  # Convert from tensor to list of 4 numbers


print("Model"+ model_num +" output:", result)

v = file_name_to_params("0450015007001")
print("\n\nDiameter (d)-->"+str(v[0])+"\nThickness (t)-->"+str(v[1])+"\nThickness (n_decimal)-->"+str(v[2])+"\n")