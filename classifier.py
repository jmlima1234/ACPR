import os
import numpy as np
from PIL import Image

def load_and_preprocess_images(data_dir,img_size):
    """
    Loads and preprocesses images from the specified directory.

    Args:
        data_dir: The path to the directory containing image folders ('wiki', 'inpainting', 'insight', 'text2img').

    Returns:
        A tuple containing:
            - X: A NumPy array of preprocessed images.
            - y: A NumPy array of corresponding labels.
                    (0 for 'wiki', 1 for 'inpainting', 2 for 'insight', 3 for 'text2img').
    """

    X = np.array([])
    y = np.array([])
    
    for label, main_folder in enumerate(['wiki', 'inpainting', 'insight', 'text2img']):
        main_folder_path = os.path.join(data_dir, main_folder)
        for subfolder_name in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(subfolder_path, filename)
                        img = Image.open(img_path)
                        img = img.resize(img_size)
                        img_array = np.array(img) / 255.0
                        X.append(img_array)
                        y.append(label)

    return np.array(X), np.array(y)

# Example usage
data_directory = 'path/to/your/dataset'  # Replace with the actual path
img_size = (64,64) # Need to adjust acording the model input size

X, y = load_and_preprocess_images(data_directory,img_size)