import argparse
import warnings
from collections import OrderedDict
from typing import Optional

import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

import utils  # Assuming utils has the functions for loading datasets and models

warnings.filterwarnings("ignore")


class Client(fl.client.NumPyClient):
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        model_str: str
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if model_str == "alexnet":
            self.model = utils.load_alexnet(classes=10)
        else:
            self.model = utils.load_efficientnet(classes=10)

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
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using local data
        results = utils.train(self.model, self.train_loader, self.val_loader, epochs, self.device)

        # Return updated parameters and training results
        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(self.train_loader.dataset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        # Update model parameters with the ones received from the server
        self.set_parameters(parameters)

        # Get evaluation config
        steps: int = config["val_steps"]

        # Evaluate the model
        loss, accuracy = utils.test(self.model, self.test_loader, steps, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Flower Client")
    
    # Add client ID argument
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        help="The client ID for this instance of the client (for partitioning purposes)."
    )
    
    # Add toy argument
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Use only a small dataset for quick testing."
    )

    # Add data directory argument
    parser.add_argument(
        "--data-dir", type=str, default="/Users/harrydo/Documents/UTS/Spring24/Ilab/cifar10/cifar10",
        help="Directory where your custom CIFAR-10 dataset is stored",
    )
    
    # Add CUDA argument
    parser.add_argument(
        "--use_cuda", action="store_true", help="Use GPU if available"
    )

    # Add model choice argument
    parser.add_argument(
        "--model", type=str, default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Model architecture to use (EfficientNet or AlexNet)"
    )

    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # Load the dataset from the specified directory
    # Load the lung cancer dataset from the specified CSV and image directory
    train_loader, val_loader, test_loader = utils.load_lung_cancer_data(csv_file="/Users/harrydo/Downloads/archive-5/recurrence.csv", 
                                                                        image_dir="/Users/harrydo/Downloads/archive-5/ROI/ROI")


    if args.toy:
        # Create subsets of the dataset if the toy flag is used
        train_loader = DataLoader(Subset(train_loader.dataset, range(10)), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(val_loader.dataset, range(10)), batch_size=16)
        test_loader = DataLoader(Subset(test_loader.dataset, range(10)), batch_size=16)

    # Start the Flower client
    client = Client(train_loader, val_loader, test_loader, device, args.model)
    fl.client.start_client(server_address="172.19.113.235:8080", client=client)


if __name__ == "__main__":
    main()
