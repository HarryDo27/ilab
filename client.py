import argparse
import warnings
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import flwr as fl
import utils

warnings.filterwarnings("ignore")

class Client(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, device, model_str):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        if model_str == "alexnet":
            self.model = utils.load_alexnet(classes=10)
        elif model_str == "mobilenet":
            self.model = utils.load_mobilenet_v3(classes=10)
        else:
            self.model = utils.load_efficientnet(classes=10)
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Access configuration parameters with default values
        batch_size = config.get("batch_size", 32)
        epochs = config.get("local_epochs", 1)

        # Train the model using local data
        results = self.train(self.train_loader, self.val_loader, epochs)

        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.train_loader.dataset)

        return parameters_prime, num_examples_train, results

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for batch in train_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.evaluate_metrics(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, F1 Score: {val_f1:.4f}")
            self.scheduler.step(val_loss)

        val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate_metrics(val_loader)
        return {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        }

    def evaluate_metrics(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        precision, recall, f1 = utils.compute_precision_recall_f1(all_labels, all_preds)
        return running_loss / len(dataloader), accuracy, precision, recall, f1

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, precision, recall, f1 = self.evaluate_metrics(self.test_loader)
        return float(loss), len(self.test_loader.dataset), {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "val_loss": float(loss),
        }

def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID for partitioning purposes")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory where the dataset is stored")
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "efficientnet", "alexnet"], help="Model architecture to use")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    train_loader, val_loader, test_loader = utils.load_lung_cancer_data(
        train_csv=f"{args.data_dir}/train.csv",
        val_csv=f"{args.data_dir}/val.csv",
        test_csv=f"{args.data_dir}/test.csv",
        image_dir=f"{args.data_dir}/ROI",
        batch_size=32
    )

    client = Client(train_loader, val_loader, test_loader, device, args.model)
    fl.client.start_client(server_address="192.168.20.3:8080", client=client)

if __name__ == "__main__":
    main()
