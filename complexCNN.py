import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available with PyTorch
print("Checking for available hardware...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("No GPU found, using CPU instead")

def load_and_preprocess_images(data_dir, img_size, save_path='preprocessed_data.npz'):
    """
    Loads and preprocesses images from the specified directory.
    Saves the preprocessed data to a file to avoid reprocessing.
    """
    # Check if preprocessed data already exists
    if os.path.exists(save_path):
        print(f"Loading preprocessed data from {save_path}...")
        data = np.load(save_path)
        return data['X'], data['y']
    
    print(f"Loading and preprocessing images from {data_dir}...")
    X = []
    y = []
    
    for label, main_folder in enumerate(['wiki', 'inpainting', 'insight', 'text2img']):
        main_folder_path = os.path.join(data_dir, main_folder)
        print(f"Processing category {main_folder} (label {label})...")
        image_count = 0
        
        for subfolder_name in os.listdir(main_folder_path):
            subfolder_path = os.path.join(main_folder_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.jpg'):
                        img_path = os.path.join(subfolder_path, filename)
                        try:
                            img = Image.open(img_path).convert('RGB')  # Ensure RGB format
                            img = img.resize(img_size)
                            img_array = np.array(img) / 255.0
                            X.append(img_array)
                            y.append(label)
                            image_count += 1
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                        
        print(f"  - Loaded {image_count} images for category {main_folder}")

    print(f"Total images loaded: {len(X)}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Save preprocessed data
    print(f"Saving preprocessed data to {save_path}...")
    np.savez(save_path, X=X, y=y)
    
    return X, y

# Define a simplified CNN
class ComplexCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(ComplexCNN, self).__init__()

        self.features = nn.Sequential(
            # The number of filters (output channels) increases in each subsequent convolutional block (32 -> 64 -> 128 ->256).  This allows the network to learn increasingly complex and abstract features.

            # Block 1: Increase depth, add Batch Norm and Dropout
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Batch Normalization
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),  # Dropout

            # Block 2:  Increase depth further, add Batch Norm and Dropout
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.28),

            # Block 3:  Increase depth and complexity, add Batch Norm and Dropout
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3), # Increased dropout slightly

             # Block 4:  Further Increase Depth
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.32), # Increased dropout slightly
        )

        # Calculate the correct input size for the fully connected layers.
        # With 64x64 input and four 2x2 max pooling operations:
        # 64 / 2 / 2 / 2 / 2 = 4
        self.classifier = nn.Sequential(
            # The classifier part of the network now has two fully connected layers (nn.Linear) instead of one.  
            # This increases the model's capacity to learn non-linear relationships between the features extracted by the convolutional layers and the output classes.  
            # Batch normalization and dropout are also applied to the fully connected layers.
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),  # Increased hidden units
            nn.BatchNorm1d(1024), # Batch Normalization for 1D layers
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout for fully connected layers
            nn.Linear(1024, 512), # Added another fully connected layer
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),  # Output layer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# Main pipeline
if __name__ == "__main__":
    print("\n--- Starting Image Classification Pipeline ---")
    data_directory = './data'  # Replace with the actual path
    img_size = (64, 64)  # Set image size
    print(f"Image size set to {img_size}")

    # Load and preprocess data
    try:
        X, y = load_and_preprocess_images(data_directory, img_size)
        
        # Split the data
        print("\nSplitting data into training and testing sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Training set size: {len(X_train)} images")
        print(f"Testing set size: {len(X_test)} images")

        # Convert to PyTorch tensors with channel-first format
        print("\nConverting data to PyTorch tensors...")
        X_train = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
        X_test = torch.from_numpy(X_test.transpose(0, 3, 1, 2)).float()
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        print(f"Training data shape: {X_train.shape}")

        # Create DataLoaders
        batch_size = 32
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, loss function, and optimizer
        # Added weight decay to the Adam optimizer. 
        # This adds a small penalty to the loss function based on the magnitude of the network's weights. 
        # This is another form of regularization that helps prevent overfitting.
        model = ComplexCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True) # Add this line
        print(model)

        # Training loop
        epochs = 20
        print("\n--- Starting Training ---")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i+1) % 10 == 0:
                    print(f'  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            epoch_acc = 100 * correct / total
            print(f'  Epoch {epoch+1} completed - Accuracy: {epoch_acc:.2f}%')

        # Evaluation
        print("\n--- Starting Evaluation ---")
        model.eval()
        class_correct = [0] * 4
        class_total = [0] * 4
        class_names = ['wiki', 'inpainting', 'insight', 'text2img']
        
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        print(f'Overall test accuracy: {accuracy:.2f}%')

        print("\nPer-class accuracy:")
        for i in range(4):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"  {class_names[i]}: {class_acc:.2f}%")
            else:
                print(f"  {class_names[i]}: N/A (no test samples)")

        print("\n--- Classification Complete ---")
        
    except Exception as e:
        print(f"Error in classification pipeline: {str(e)}")
