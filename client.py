import argparse
import warnings
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import flwr as fl
import utils  # Assuming utils has the functions for loading datasets and models
from IPython import embed

warnings.filterwarnings("ignore")


class Client(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, device, model_str):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Load the model based on model_str argument
        if model_str == "alexnet":
            self.model = utils.load_alexnet(classes=10)
        elif model_str == "mobilenet":
            self.model = utils.load_mobilenet_v3(classes=10)
        else:
            self.model = utils.load_efficientnet(classes=10)
        
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    def set_parameters(self, parameters):
        """Loads a model and replaces its parameters with the ones given."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model on the local training set."""
        # Update model parameters with the ones received from the server
        self.set_parameters(parameters)

        # Get training hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        # Train the model using local data
        results = self.train(self.train_loader, self.val_loader, epochs)

        # Return updated parameters and training results
        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.train_loader.dataset)

        return parameters_prime, num_examples_train, results

    def train(self, train_loader, val_loader, num_epochs):
        """Train the model and apply the scheduler."""

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

            # After each epoch, evaluate on the validation set
            val_loss, val_accuracy = utils.test(self.model, val_loader, device=self.device)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Step the scheduler based on validation loss
            self.scheduler.step(val_loss)

        # After training, return the validation loss and accuracy
        train_loss, train_acc = utils.test(self.model, train_loader, device=self.device)
        val_loss, val_acc = utils.test(self.model, val_loader, device=self.device)

        return {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        # Update model parameters with the ones received from the server
        self.set_parameters(parameters)

        # Get evaluation config
        steps = config["val_steps"]

        # Evaluate the model
        loss, accuracy = utils.test(self.model, self.test_loader, steps, device=self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--client-id", type=int, default=0, help="Client ID for partitioning purposes")
    parser.add_argument("--toy", action="store_true", help="Use only a small dataset for quick testing.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory where the dataset is stored")
    parser.add_argument("--use_cuda", action="store_true", help="Use GPU if available")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["mobilenet", "efficientnet", "alexnet"],
                        help="Model architecture to use (Mobilenet, EfficientNet, or AlexNet)")

    args = parser.parse_args()

    # Set the device
    # device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the lung cancer dataset
    train_loader, val_loader, test_loader = utils.load_lung_cancer_data(
        train_csv=f"{args.data_dir}/train.csv",
        val_csv=f"{args.data_dir}/val.csv",
        test_csv=f"{args.data_dir}/test.csv",
        image_dir=f"{args.data_dir}/ROI",
        batch_size=32
    )

    # If toy mode is enabled, use a smaller subset of data
    if args.toy:
        train_loader = DataLoader(Subset(train_loader.dataset, range(10)), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(val_loader.dataset, range(10)), batch_size=16)
        test_loader = DataLoader(Subset(test_loader.dataset, range(10)), batch_size=16)

    # Initialize the client and start Flower
    client = Client(train_loader, val_loader, test_loader, device, args.model)
    fl.client.start_client(server_address="192.168.55.111:8080", client=client)


if __name__ == "__main__":
    main()
