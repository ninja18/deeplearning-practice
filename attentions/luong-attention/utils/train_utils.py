import json
import math
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchmetrics.text import BLEUScore
from utils.multi30k_data_processing_utils import decode, decode_tokens


def save_artifacts(
    model,
    train_losses,
    val_losses,
    val_bleu_scores,
    teacher_forcing_ratios,
    batch_size,
    learning_rate,
    epochs,
    artifacts_path,
    model_path,
    max_grad_norm,
):
    with open(artifacts_path, "w") as f:
        artifacts = {
            "train_losses": train_losses,
            "val_losses": val_losses,
         'val_bleu_scores': val_bleu_scores,
         'teacher_forcing_ratios': teacher_forcing_ratios,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "model": str(model),
            "model_path": model_path,
            "max_grad_norm": max_grad_norm,
        }
        json.dump(artifacts, f)


def show_graph(train_losses, val_losses, val_bleu_scores, save_path=None, save=False):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.plot(train_losses, label='Training loss')
    ax1.plot(val_losses, label='Validation loss')
    ax1.legend()
    ax1.set_title("Loss over epochs")

    ax2.plot(val_bleu_scores, label='Validation BLEU')
    ax2.legend()
    ax2.set_title("BLEU over epochs")
    fig.show()
    if save and save_path:
        fig.savefig(save_path)

def inverse_sigmoid_decay(step, k=8):
    """
    Calculates the teacher forcing ratio using inverse sigmoid decay.
    Formula: k / (k + exp(step / k))

    Args:
        step: Current epoch or iteration number (0-indexed).
        k: Decay constant. Larger k means slower decay.
    """
    ratio = k / (k + math.exp(step / k))
    return round(ratio, 2)

def batch_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time / 60)
    seconds = int(elapsed_time % 60)
    return minutes, seconds


def load_model(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def train(model, loader, criterion, optimizer, max_grad_norm, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()

        logits_all = model(
            batch["source"], batch["source_lengths"], batch["target"], teacher_forcing_ratio=teacher_forcing_ratio
        )
        target = batch["target"][:, 1:].reshape(-1)

        loss = criterion(logits_all.reshape(-1, logits_all.shape[-1]), target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def validate_with_teacher_forcing(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        logits_all = model(
            batch["source"], batch["source_lengths"], batch["target"], teacher_forcing_ratio=1.0
        )
        target = batch["target"][:, 1:].reshape(-1)

        loss = criterion(logits_all.reshape(-1, logits_all.shape[-1]), target)
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


@torch.no_grad()
def validate_with_bleu_score(model, loader, index_to_vocab, device):
    model.eval()
    metric = BLEUScore()

    preds_text = []
    targets_text = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        src = batch["source"]
        src_lengths = batch["source_lengths"]
        trg = batch["target"]

        prediction, _ = model.generate(src, src_lengths)

        for i in range(src.shape[0]):
            target_str = decode(index_to_vocab, trg[i].tolist())
            targets_text.append([target_str]) # Expects list of references per sample

            pred_str = decode(index_to_vocab, prediction[i].tolist())
            preds_text.append(pred_str)

    score = metric(preds_text, targets_text)
    return score.item()

def display_attention(sentence, translation, attention, n_cols=2, figure_size=(10, 10)):
    """
    Plots the attention weights matrix.

    Args:
        sentence (list[str]): Source tokens.
        translation (list[str]): Predicted tokens.
        attention (torch.Tensor or np.array): Shape (len(translation), len(sentence)).
    """
    # Ensure attention is a numpy array and on CPU
    if hasattr(attention, "cpu"):
        attention = attention.squeeze().cpu().detach().numpy()

    # Setup plot
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)

    # Plot heatmap
    cax = ax.matshow(attention, cmap="bone")
    fig.colorbar(cax)

    # Set axes
    ax.tick_params(labelsize=12)

    # Set ticks positions
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set labels
    # Note: [''] + ... is often used because matplotlib might offset labels by 1 in some versions/configs,
    # but with MultipleLocator(1) starting at 0, exact lists usually work best.
    # If alignment is off, try prepending an empty string: [''] + sentence
    ax.set_xticklabels([""] + sentence, rotation=90)
    ax.set_yticklabels([""] + translation)

    # Show label at every tick
    ax.grid(False)

    plt.show()
    plt.close()



@torch.no_grad()
def translate(model, batch, source_index_to_vocab, target_index_to_vocab, topk, device):
    model.eval()

    batch = batch_to_device(batch, device)

    targets = batch["target"]
    preds_all, attention_weights_all = model.generate(batch["source"], batch["source_lengths"], topk=topk)

    translation_results = {}

    for i in range(preds_all.shape[0]):
        source_sentence = decode_tokens(source_index_to_vocab, batch['source'][i].tolist())
        target_sentence = decode_tokens(target_index_to_vocab, targets[i].tolist())
        attention_weights = attention_weights_all[i]

        prediction = decode_tokens(target_index_to_vocab, preds_all[i].tolist())

        translation_results[i] = {
            "source": source_sentence,
            "prediction": prediction,
            "target": target_sentence,
            "attention_weights": attention_weights
        }

        print(f"Source: {" ".join(source_sentence)}")
        print(f"Pred: {" ".join(prediction)}")
        print(f"Target: {" ".join(target_sentence)}")
        print("-"*100)
    return translation_results