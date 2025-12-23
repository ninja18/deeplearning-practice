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

def batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
     for k, v in batch.items()}

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds

def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def train(model, loader, criterion, optimizer, max_grad_norm, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()

        logits_all, preds_all, _ = model(batch["source"], batch["source_lengths"], batch["target"])
        target = batch["target"][:, 1:].reshape(-1)

        loss = criterion(logits_all.reshape(-1, logits_all.shape[-1]), target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits_all, preds_all, _ = model(batch["source"], batch["source_lengths"], batch["target"])
        target = batch["target"][:, 1:].reshape(-1)

        loss = criterion(logits_all.reshape(-1, logits_all.shape[-1]), target)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)