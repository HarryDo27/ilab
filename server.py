import argparse
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import flwr as fl
import torch
import utils

warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 16, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return None as no centralized evaluation is needed."""
    return None


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self, rnd: int, results, failures
    ):
        """Aggregate fit results and print client results."""
        print(f"Round {rnd} - Results from clients:")
        
        # Aggregate loss and accuracy across all clients
        total_loss = 0
        total_accuracy = 0
        total_samples = 0
        
        for i, (client, fit_res) in enumerate(results):
            # Extract client-side metrics
            val_loss = fit_res.metrics.get("val_loss")
            val_accuracy = fit_res.metrics.get("val_accuracy")
            num_examples = fit_res.num_examples
            
            print(f"Client {i} - Loss: {val_loss}, Accuracy: {val_accuracy}")
            
            # Aggregate results across clients
            total_loss += val_loss * num_examples
            total_accuracy += val_accuracy * num_examples
            total_samples += num_examples
        
        # Calculate average loss and accuracy across all clients
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        
        print(f"Aggregated - Loss: {avg_loss}, Accuracy: {avg_accuracy}")
        
        # Return the aggregated loss and accuracy
        return super().aggregate_fit(rnd, results, failures)


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either EfficientNet or AlexNet models. \
             If you want to achieve differential privacy, please use the AlexNet model",
    )

    args = parser.parse_args()

    # Load the selected model
    if args.model == "alexnet":
        model = utils.load_alexnet(classes=10)
    else:
        model = utils.load_efficientnet(classes=10)

    # Initialize model parameters
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    print('Initializing strategy')

    # Create the FedAvg strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters)
    )

    print('Starting server')

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="192.168.20.3:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )

    print('Finished setting up server')


if __name__ == "__main__":
    main()
