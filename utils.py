import torch
import numpy as np
import random
from typing import List, Iterator
import torch.nn as nn
from models.moe import CNN, CNN_moe_noise
import torch.optim as optim
from engine.base_trainer import BaseCVTrainer, BaseRegressionTrainer, BaseTrainer
from engine.fedprox_trainer import FedProxCVTrainer
from tqdm import tqdm

# model list
MODEL_LIST = {
    "cnn": CNN,
    "cnn_moe": CNN_moe_noise
}

# trainer list (nested dictionary)
# we only consider CV or NLP task
TRAINER_LIST = {
    "CV": {
        "FedAvg": BaseCVTrainer,
        "FedFT": BaseCVTrainer,
        "FedAvgDense": BaseCVTrainer,
        "FedProx": FedProxCVTrainer,
        "FedRA": BaseCVTrainer,
        "FedBN": BaseCVTrainer,
        "pFedMe": BaseCVTrainer,
        "FedHist": BaseCVTrainer,
        "Standalone": BaseCVTrainer,
    },
    "Regression": {
        "FedAvg": BaseRegressionTrainer,
        "FedFT": BaseRegressionTrainer,
        "FedAvgDense": BaseRegressionTrainer,
        "FedProx": BaseRegressionTrainer,  # need modification
        "FedRA": BaseRegressionTrainer,  # need modification
        "FedBN": BaseRegressionTrainer,
        "pFedMe": BaseRegressionTrainer,
        "FedHist": BaseRegressionTrainer,
        "Standalone": BaseRegressionTrainer,
    },
}

# utility functions
def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_model(model_name: str, model_kwargs: dict) -> nn.Module:
    """build model according to the given model_name

    Args:
        model_name (str): model name
        model_kwargs (dict): the model keyword arguments
    """
    model: nn.Module = MODEL_LIST[model_name](**model_kwargs)
    return model


def build_optimizer(
    optimizer_name: str, parameters: Iterator[nn.Parameter], optimizer_kwargs: dict
) -> optim.Optimizer:
    """build optimizer according to the given parameters

    Args:
        optimizer_name (str): optimizer name
        parameters (Iterator[nn.Parameter]): the parameters need for optimization
        optimizer_kwargs (dict): optimzier keyword arguments (such as lr and weight_decay)

    Returns:
        optim.Optimizer: the given optimizer
    """
    optimizer: optim.Optimizer = getattr(optim, optimizer_name)(
        params=parameters, **optimizer_kwargs
    )
    return optimizer


def build_trainer(task: str, algorithm: str) -> BaseTrainer:
    """build a trainer according to task and algorithm

    Args:
        task (str): -
        algorithm (str): -
    Returns:
        BaseTrainer: the trainer
    """
    return TRAINER_LIST[task][algorithm]()


def build_criterion(criterion: str) -> nn.Module:
    """build a criterion according the criterion name

    Args:
        criterion (str): criterion

    Returns:
        nn.Module: criterion
    """
    return getattr(nn, criterion)()


def get_parameters(net: nn.Module) -> List[np.ndarray]:
    """get parameters (ndarray format) of a network

    Args:
        net (nn.Module): the given network

    Returns:
        List[np.ndarray]: the returned parameters
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Training Loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, gating_loss = model(images)
        loss = criterion(outputs, labels)
        if gating_loss is not None:
            loss += gating_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# Evaluation Loop
def evaluate(model, test_loader, criterion, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    running_loss = 0.0
    progress_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs, gating_loss = model(images)
            loss = criterion(outputs, labels)
            if gating_loss is not None:
                loss += gating_loss
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy