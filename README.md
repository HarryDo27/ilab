# Federated Learning for Lung Cancer Recurrence

A privacy-preserving **federated learning system** built with **PyTorch** and **Flower** to predict recurrence of Non-Small Cell Lung Cancer (NSCLC) from CT scans and clinical data. Combines **CNN architectures** (ResNet, MobileNet, EfficientNet) with federated averaging to enable collaborative training across hospitals without sharing sensitive data.

---

## Features
- **Federated Learning Setup**: Multiple clients train locally and share only model weights with a central server.  
- **Deep Learning Models**: Pretrained CNNs fine-tuned for medical imaging.  
- **Privacy-Preserving**: GDPR-compliant federated aggregation (FedAvg).  
- **Healthcare Focus**: Optimized for recall and model transparency in clinical use cases.  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/HarryDo27/lungcancer-fl.git
   cd lungcancer-fl
2. Run the requirement:
    ```bash
    pip install -r requirements.txt
3. Update server IP in server.py and client.py
4. Run the system
    ```bash
    ./run.sh

---
## Highlights
- Enables collaborative training across multiple clients without sharing raw data.
- Achieved competitive recall and accuracy while maintaining patient data privacy.
- Demonstrates real-world application of federated learning in healthcare.

---
## Result
- **Centralized model**: Recall ~0.60, Accuracy ~0.66
- **Federated model**: Recall ~0.59, Accuracy ~0.56
- Achieved competitive performance while maintaining patient data privacy.


