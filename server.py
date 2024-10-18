import argparse
import warnings
import flwr as fl
import torch
import utils
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,  # Ensure that batch_size is defined
        "local_epochs": 10,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_f1_score = 0
        self.best_parameters = None
        self.history = []
        self.best_round = None

    def aggregate_fit(self, rnd, results, failures):
        print(f"Round {rnd} - Results from clients:")
        
        total_loss, total_accuracy, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0
        total_samples = 0

        # Collect metrics from all clients
        for i, (client, fit_res) in enumerate(results):
            val_loss = fit_res.metrics.get("val_loss", 0.0)
            val_accuracy = fit_res.metrics.get("val_accuracy", 0.0)
            val_precision = fit_res.metrics.get("val_precision", 0.0)
            val_recall = fit_res.metrics.get("val_recall", 0.0)
            val_f1 = fit_res.metrics.get("val_f1", 0.0)
            num_examples = fit_res.num_examples

            print(f"Client {i} - Loss: {val_loss}, Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}, F1 Score: {val_f1}")

            total_loss += val_loss * num_examples
            total_accuracy += val_accuracy * num_examples
            total_precision += val_precision * num_examples
            total_recall += val_recall * num_examples
            total_f1 += val_f1 * num_examples
            total_samples += num_examples

        # Compute averages for the round
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        avg_precision = total_precision / total_samples if total_samples > 0 else 0
        avg_recall = total_recall / total_samples if total_samples > 0 else 0
        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0

        print(f"Aggregated - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")
        
        # Save the best model based on F1 score
        if avg_f1 > self.best_f1_score:
            self.best_f1_score = avg_f1
            self.best_parameters = results[0][1].parameters
            self.best_round = rnd

        # Save metrics to history for visualization later
        self.history.append({
            "round": rnd,
            "loss": avg_loss,
            "accuracy": avg_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1
        })

        return super().aggregate_fit(rnd, results, failures)

    def on_conclude(self):
        """Print a summary of metrics collected over all rounds and visualize them."""
        print("\n[FINAL SUMMARY]")
        for record in self.history:
            print(f"Round {record['round']} - Loss: {record['loss']:.4f}, Accuracy: {record['accuracy']:.4f}, "
                  f"Precision: {record['precision']:.4f}, Recall: {record['recall']:.4f}, F1 Score: {record['f1_score']:.4f}")
        
        # Plotting loss and metrics
        rounds = [record["round"] for record in self.history]
        losses = [record["loss"] for record in self.history]
        accuracies = [record["accuracy"] for record in self.history]
        precisions = [record["precision"] for record in self.history]
        recalls = [record["recall"] for record in self.history]
        f1_scores = [record["f1_score"] for record in self.history]

        plt.figure(figsize=(12, 8))
        
        # Plot average loss curve
        plt.subplot(2, 1, 1)
        plt.plot(rounds, losses, marker='o', label='Average Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Average Loss per Round')
        plt.grid(True)

        # Plot average metrics
        plt.subplot(2, 1, 2)
        plt.plot(rounds, accuracies, marker='o', label='Accuracy')
        plt.plot(rounds, precisions, marker='o', label='Precision')
        plt.plot(rounds, recalls, marker='o', label='Recall')
        plt.plot(rounds, f1_scores, marker='o', label='F1 Score')
        plt.xlabel('Round')
        plt.ylabel('Metrics')
        plt.title('Metrics per Round')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def evaluate_best_on_test(self, test_loader):
        """Evaluate the best model based on validation F1 score on the test set."""
        if self.best_parameters is None:
            print("No best model found.")
            return
        
        # Create the model and set its parameters
        model = utils.load_mobilenet_v3(classes=10)  # Adjust based on your architecture
        model.load_state_dict(torch.load(self.best_parameters))
        model.eval()
        
        # Evaluate on the test set
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for images, labels in test_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
        
        # Compute metrics
        precision, recall, f1 = utils.compute_precision_recall_f1(all_labels, all_preds)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        
        print(f"Best Model (from Round {self.best_round}) - Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["efficientnet", "alexnet", "mobilenet"], help="Model architecture to use")

    args = parser.parse_args()
    model = utils.load_mobilenet_v3(classes=10) if args.model == "mobilenet" else utils.load_efficientnet(classes=10)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters)
    )

    fl.server.start_server(
        server_address="192.168.20.3:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )

    # Print final summary and visualize
    strategy.on_conclude()

if __name__ == "__main__":
    main()
