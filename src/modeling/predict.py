import torch
import numpy as np
from tqdm import tqdm
from src.utils import calculate_metric

def predict(dataloader, model, cfg):
    # Validation mode
    model.eval()

    final_y = []
    final_y_pred = []

    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg.device)
        y = batch[1].to(cfg.device)

        with torch.no_grad():
            # Forward: Get model outputs
            y_pred = model(X)


            # Covert y and y_pred to lists
            y =  y.detach().cpu().numpy().tolist()
            y_pred =  y_pred.detach().cpu().numpy().tolist()

            # Extend original list
            final_y.extend(y)
            final_y_pred.extend(y_pred)

    # Calculate statistics
    final_y_pred_argmax = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred_argmax)
    return final_y_pred_argmax, metric
