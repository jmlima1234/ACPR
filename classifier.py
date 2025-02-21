import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

# Split the data into training and testing sets for now
# This will be replaced with a more sophisticated method in the future
# We need to investigate how to optimize. It is a good idea acording to the project proposal
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(4, activation='softmax')  # Adjusted for 4 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Adjusted for 4 classes
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

# Simple metrics because it is a simple and initial implementation
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

