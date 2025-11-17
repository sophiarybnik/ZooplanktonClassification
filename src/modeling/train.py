import torch
from src.utils import calculate_metric, set_seed
import numpy as np
from tqdm import tqdm


def train_one_epoch(dataloader, model, optimizer, scheduler, cfg, lrs):
    # Training mode
    model.train()
    
    # Init lists to store y and y_pred
    final_y = []
    final_y_pred = []
    final_loss = []
    
    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg.device)
        y = batch[1].to(cfg.device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Forward: Get model outputs
            y_pred = model(X)
            
            # Forward: Calculate loss
            loss = cfg.criterion(y_pred, y)
            
            # Convert y and y_pred to lists
            y =  y.detach().cpu().numpy().tolist()
            y_pred =  y_pred.detach().cpu().numpy().tolist()
            
            # Extend original list
            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

            # Backward: Optimize
            loss.backward()
            optimizer.step()
            
                    
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        
    # Calculate statistics
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)
        
    return metric, loss, lrs


def validate_one_epoch(dataloader, model, cfg):
    # Validation mode
    model.eval()
    
    final_y = []
    final_y_pred = []
    final_loss = []
    
    # Iterate over data
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        X = batch[0].to(cfg.device)
        y = batch[1].to(cfg.device)

        with torch.no_grad():
            # Forward: Get model outputs
            y_pred = model(X)
            
            # Forward: Calculate loss
            loss = cfg.criterion(y_pred, y)  

            # Covert y and y_pred to lists
            y =  y.detach().cpu().numpy().tolist()
            y_pred =  y_pred.detach().cpu().numpy().tolist()
            
            # Extend original list
            final_y.extend(y)
            final_y_pred.extend(y_pred)
            final_loss.append(loss.item())

    # Calculate statistics
    loss = np.mean(final_loss)
    final_y_pred = np.argmax(final_y_pred, axis=1)
    metric = calculate_metric(final_y, final_y_pred)
        
    return metric, loss


def fit(model, optimizer, scheduler, cfg, train_dataloader, valid_dataloader=None):
    lrs = []

    acc_list = []
    loss_list = []
    val_acc_list = []
    val_loss_list = []

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch + 1}/{cfg.epochs}")

        set_seed(cfg.seed + epoch)

        acc, loss, lrs = train_one_epoch(train_dataloader, model, optimizer, scheduler, cfg, lrs)

        if valid_dataloader:
            val_acc, val_loss = validate_one_epoch(valid_dataloader, model, cfg)

        print(f'Train Loss: {loss:.4f} Train Acc: {acc:.4f}')
        acc_list.append(acc)
        loss_list.append(loss)
        
        if valid_dataloader:
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
    
    return acc_list, loss_list, val_acc_list, val_loss_list, model, lrs