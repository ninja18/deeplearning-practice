import json
import torch
import matplotlib.pyplot as plt

def save_artifacts(model, train_losses, val_losses, batch_size, learning_rate, epochs, artifacts_path, model_path):
    with open(artifacts_path, 'w') as f:
        artifacts = {
        'train_losses': train_losses,
         'val_losses': val_losses,
         'batch_size': batch_size,
         'learning_rate': learning_rate,
         'epochs': epochs,
         'model': str(model),
         'model_path': model_path,
         }
        json.dump(artifacts, f)

def show_graph(train_losses, val_losses, save_path=None, save=False):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(train_losses, label='Training loss')
    ax1.plot(val_losses, label='Validation loss')
    ax1.legend()
    ax1.set_title("Loss over epochs")
    fig.show()
    if save and save_path:
        fig.savefig(save_path)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model