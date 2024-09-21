import warnings

import torch
from flwr_datasets import FederatedDataset
from torchvision.models import AlexNet, efficientnet_b0
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from PIL import Image

class LungCancerDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with SOPInstanceUID and labels.
            image_dir (str): Directory with all the images (in .npy format).
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Filter rows in data_frame where SOPInstanceUID matches files in the image directory
        available_images = [f.replace('.npy', '') for f in os.listdir(image_dir) if f.endswith('.npy')]
        self.data_frame = self.data_frame[self.data_frame['SOPInstanceUID'].isin(available_images)]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load the .npy file using SOPInstanceUID as the filename
        sop_uid = self.data_frame.iloc[idx]['SOPInstanceUID']
        img_name = os.path.join(self.image_dir, sop_uid + ".npy")
        image = np.load(img_name)  # Load the .npy file as a NumPy array

        # Convert the NumPy array to a PIL image (or leave it as a tensor if you prefer)
        image = Image.fromarray(image)

            # Convert 1-channel grayscale to 3-channel by repeating the array along the third axis
        image = image.convert("RGB")

    # Apply any transforms, if provided
        if self.transform:
            image = self.transform(image)

        # Get the label (Recurrence: 'yes'/'no' in the 'Recurrence' column)
        label = self.data_frame.iloc[idx]['Recurrence']
        label = 1 if label == 'yes' else 0  # Convert 'yes' to 1 and 'no' to 0

        return image, label

def load_lung_cancer_data(csv_file: str, image_dir: str, batch_size: int = 32):
    """Load lung cancer recurrence dataset from the specified CSV and image directory."""

    # Define the transformations (resize and normalize based on the needs of your dataset)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust this size as per your model's input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjust normalization if needed
    ])

    # Load the dataset
    dataset = LungCancerDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)

    # Split the dataset (e.g., 70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader



def train(
    net, trainloader, valloader, epochs, device: torch.device = torch.device("cpu")
):
    """Train the network on the training set."""
    print("Starting training...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:  # No need for batch["img"] / batch["label"]
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    # Compute train and validation loss/accuracy after training
    train_loss, train_acc = test(net, trainloader, device=device)
    val_loss, val_acc = test(net, valloader, device=device)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results



def test(
    net, testloader, steps: int = None, device: torch.device = torch.device("cpu")
):
    """Validate the network on the test set."""
    print("Starting evaluation...")
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):  # Tuple unpacking
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
    accuracy = correct / len(testloader.dataset)
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


def load_efficientnet(classes: int = 10):
    """Loads EfficientNetB0 from TorchVision."""
    efficientnet = efficientnet_b0(pretrained=True)
    # Adjust the classifier layer for the correct number of output classes
    model_classes = efficientnet.classifier[1].in_features
    if classes != model_classes:
        efficientnet.classifier[1] = torch.nn.Linear(model_classes, classes)
    return efficientnet


def load_alexnet(classes: int = 10):
    """Load AlexNet model from TorchVision."""
    alexnet = AlexNet(num_classes=classes)
    return alexnet


def get_model_params(model):
    """Returns a model's parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]