from pathlib import Path
import torch
import torch.nn as nn

class CFG:
    # Paths
    CONFIG_FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = CONFIG_FILE_PATH.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "models"


    # Dataset
    n_classes = 2 # we are predicting two classes
    use_augmentation = True # augment the training data

    # Model
    backbone = "resnet18" # 18-layer CNN with residual connections
    pretrained = True # we want the weights of the pre-trained model


    # Define Hyperparameter search space
    hp_search = {
        "image_size" : [128, 256,512],
        "batch_size": [32, 64],
        "weight_decay": [0.01, 0.001, 0.0001, 0.0],
        "epochs": [5],
        "n_folds":[3],
        "optimizer_type": ["Adam", "SGD"],
        "lr_adam": [1e-3, 5e-4, 1e-4],
        "lr_sgd": [1e-2, 5e-3, 1e-1],
        "eps": [1e-6, 1e-7, 1e-8],
        "momentum": [0.5, 0.75, 0.9, 0.999],
        "betas": [(0.8,0.9), (0.9,0.999)],
        }

    seed = 42
    criterion = nn.CrossEntropyLoss()


    # Resources
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set to gpu if available, otherwise cpu
    num_workers = 4

cfg = CFG()
