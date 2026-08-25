"""Petlja treniranja sa ranim zaustavljanjem.

Petlja prima model kao argument, pa su RNN, LSTM i GRU tri poziva iste
funkcije. Isto vazi za ocenjivanje: sigmoid se
primenjuje na jednom mestu, u predict_proba.
"""

import copy

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import MODELS_PATH, RANDOM_STATE


def set_seed(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(model, loader_train, loader_valid, epochs=100, patience=5, learning_rate=1e-3):
    """Trenira model i vraca istoriju ucenja."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [], "valid_loss": [],
        "train_accuracy": [], "valid_accuracy": [],
        "grad_norm": [],
    }
    best_valid_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_progress = 0

    for _ in range(epochs):
        train_loss, train_accuracy, max_grad_norm = _run_epoch(
            model, loader_train, criterion, optimizer
        )
        valid_loss, valid_accuracy, _ = _run_epoch(model, loader_valid, criterion)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)
        history["grad_norm"].append(max_grad_norm)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_progress = 0
        else:
            epochs_without_progress += 1
            if epochs_without_progress == patience:
                break

    model.load_state_dict(best_state)
    return history


def _run_epoch(model, loader, criterion, optimizer=None):
    """Jedan prolaz kroz loader; sa optimizerom trenira, bez njega ocenjuje.

    Gubitak i tacnost se racunaju kao prosek otezan velicinom paketica, jer
    poslednji paketic u epohi nije pun. Vraca (gubitak, tacnost, najveca norma
    gradijenta); bez optimizera je norma 0.0.
    """
    is_training = optimizer is not None
    model.train(is_training)

    total_loss, n_correct, n_examples = 0.0, 0, 0
    max_grad_norm = 0.0

    with torch.set_grad_enabled(is_training):
        for seq_home, seq_away, context, labels in loader:
            logits = model(seq_home, seq_away, context)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                max_grad_norm = max(max_grad_norm, _gradient_norm(model))
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            n_correct += ((logits > 0) == (labels > 0.5)).sum().item()
            n_examples += batch_size

    return total_loss / n_examples, n_correct / n_examples, max_grad_norm


def _gradient_norm(model):
    """Vraca ukupnu L2 normu gradijenta preko svih parametara."""
    # meri se, ne odseca - na ovom skupu norma ne prelazi 1.0, pa bi
    # clip_grad_norm_ bio prag koji se nikad ne dostigne
    squares = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    return torch.sqrt(torch.stack(squares).sum()).item()


def predict_proba(model, loader):
    """Vraca verovatnoce pobede domacina i stvarne oznake, kao numpy nizove.

    Model vraca sirove logite, a evaluate.compute_metrics ocekuje
    verovatnoce, pa se sigmoid primenjuje ovde - na jednom mestu, umesto u
    svakoj svesci ponovo.
    """
    model.eval()
    probabilities, targets = [], []

    with torch.no_grad():
        for seq_home, seq_away, context, labels in loader:
            probabilities.append(torch.sigmoid(model(seq_home, seq_away, context)))
            targets.append(labels)

    return torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def save_model(model, file_name):
    """Cuva state_dict modela u models/, ne ceo objekat.

    state_dict cuva samo tezine, pa se model moze ucitati i posle izmene koda
    klase. torch.save nad celim objektom cuva i putanju do klase i puca ako se
    modul preimenuje.
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_PATH / file_name)
