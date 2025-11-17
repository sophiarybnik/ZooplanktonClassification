import os
import numpy as np
import torch
import random
from sklearn.metrics import accuracy_score


def calculate_metric(y, y_pred):
    metric = accuracy_score(y, y_pred)
    return metric

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA on 1 GPU
    torch.cuda.manual_seed(seed)
    
    # If using CUDA on more than 1 GPU
    torch.cuda.manual_seed_all(seed)
    
    # Make certain operations deterministic
    torch.backends.cudnn.deterministic = True 
    
    # Disable auto-tuner for reproducibility
    torch.backends.cudnn.benchmark = False


def sample_hyperparameters(cfg):
    sampled = {}
    for key, values in cfg.hp_search.items():
        sampled[key] = random.choice(values)
    return sampled


def apply_hyperparameters(cfg, sampled):

    # Shared
    cfg.image_size = sampled["image_size"]
    cfg.batch_size  = sampled["batch_size"]
    cfg.weight_decay = sampled["weight_decay"]
    cfg.epochs = sampled["epochs"]
    cfg.n_folds = sampled["n_folds"]

    # Optimizer
    cfg.optimizer_type = sampled["optimizer_type"]

    if cfg.optimizer_type == "Adam":
        cfg.learning_rate = sampled["lr_adam"]
        cfg.eps = sampled["eps"]
        cfg.betas = sampled["betas"]
        cfg.momentum = None     # not used
    
    else:  # SGD
        cfg.learning_rate = sampled["lr_sgd"]
        cfg.momentum = sampled["momentum"]
        cfg.betas = None        # not used
        cfg.eps = None          # not used

    return cfg