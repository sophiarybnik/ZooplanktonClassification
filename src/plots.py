import matplotlib.pyplot as plt
def visualize_history(cfg, fold_train_loss, fold_val_loss):
    fig, ax = plt.subplots(1,cfg.n_folds, figsize=(12,4))
    for i in range(cfg.n_folds):
        train_loss = fold_train_loss[i]
        val_loss = fold_val_loss[i]
        ax[i].plot(range(1,cfg.epochs+1), train_loss,  color='darkgrey', label = 'training')
        ax[i].plot(range(1,cfg.epochs+1), val_loss,  color='cornflowerblue', label = 'validation')
        ax[i].set_title('Loss')
        ax[i].set_xlabel('Epochs')
        ax[i].legend(loc="upper right")

    plt.show()     
    return