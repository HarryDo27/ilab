import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torchvision.models import AlexNet, efficientnet_b0, mobilenet_v3_large
from sklearn.metrics import f1_score, precision_score, recall_score

# Dataset Class
class LungCancerDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sop_uid = self.data_frame.iloc[idx]['SOPInstanceUID']

        # Find the corresponding image file (with _1, _2, etc. suffixes)
        found_image = None
        for suffix in range(1, 10):
            img_name = os.path.join(self.image_dir, f"{sop_uid}_{suffix}.tiff")
            if os.path.exists(img_name):
                found_image = img_name
                break

        if found_image is None:
            raise FileNotFoundError(f"No image found for {sop_uid}")

        image = Image.open(found_image).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx]['Recurrence']
        label = 1 if label == 'yes' else 0

        return image, label

# Load dataset
def load_lung_cancer_data(train_csv, val_csv, test_csv, image_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = LungCancerDataset(csv_file=train_csv, image_dir=image_dir, transform=transform)
    val_set = LungCancerDataset(csv_file=val_csv, image_dir=image_dir, transform=transform)
    test_set = LungCancerDataset(csv_file=test_csv, image_dir=image_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

# Model Loaders
def load_alexnet(classes=10):
    model = AlexNet(num_classes=classes)
    return model

def load_efficientnet(classes=10):
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, classes)
    return model

def load_mobilenet_v3(classes=10):
    model = mobilenet_v3_large(pretrained=True)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, classes)
    return model

# Metric Computation
def compute_precision_recall_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

# Extract Model Parameters
def get_model_params(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
