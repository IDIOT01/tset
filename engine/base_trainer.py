from typing import Dict, Iterable
import torch
import torch.nn as nn
from torch import optim
from timm.utils import accuracy
import numpy as np
import sys
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
    ) -> float:
        """Train the model on the given data_loader for E epoch.

        Args:
            model (nn.Module): model
            criterion (nn.Module): loss function
            data_loader (Iterable): the given train loader
            optimizer (optim.Optimizer): optimizer to use for training
            epoch (int): how many epochs to train
            device (torch.device): which device to train
        Returns:
            the average training loss of the E epochs
        """
        pass

    @abstractmethod
    def evalulate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        device: torch.device,
    ) -> tuple[float, Dict]:
        """Evaluate the model on the given data_loader.

        Args:
            model (nn.Module): the model
            criterion (nn.Module): loss function
            data_loader (Iterable): the test loader
            device (torch.device): which device to evaluate
        Returns:
            the test loss and a dictionary containing the test metrics.
        """
        pass


class BaseCVTrainer(BaseTrainer):

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
    ) -> float:
        """Train the model on the given data_loader for E epoch.

        Args:
            model (nn.Module): model
            criterion (nn.Module): loss function
            data_loader (Iterable): the given train loader
            optimizer (optim.Optimizer): optimizer to use for training
            epoch (int): how many epochs to train
            device (torch.device): which device to train
        Returns:
            the average training loss of the E epochs
        """
        # Switch to training mode
        model.to(device)
        model.train()
        losses = []
        # Start the main loop
        for ep in range(epoch):
            # Start the inner loop (batch)
            for samples, targets in data_loader:
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                # use auto mixed precision to accerate the training procedure.
                with torch.cuda.amp.autocast():
                    outputs, _ = model.forward(samples)
                    loss: torch.Tensor = criterion.forward(outputs, targets)
                # If the loss value is infinite, stop
                loss_value = loss.item()
                if not np.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                # optimizer cleans up and step
                optimizer.zero_grad()
                # torch.cuda.synchronize()
                loss.backward()

                optimizer.step()

                # Record the loss
                losses.append(loss_value)
        return np.mean(losses)

    @torch.no_grad()
    def evalulate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        device: torch.device,
    ) -> tuple[float, Dict]:
        """Evaluate the model on the given data_loader.

        Args:
            model (nn.Module): the model
            criterion (nn.Module): loss function
            data_loader (Iterable): the test loader
            device (torch.device): which device to evaluate
        Returns:
            the test loss and a dictionary containing the training acc and other metrics.
        """
        # switch to eval model
        model.to(device)
        model.eval()

        losses = []
        results: dict = {
            "acc_top5": [],
            "acc_top1": [],
        }
        # start the main loop
        for images, target in data_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # comput the outputs
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
            # calculate the top1 and top5 accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_value = loss.item()
            # log
            losses.append(loss_value)
            results["acc_top1"].append(acc1.item())
            results["acc_top5"].append(acc5.item())

        loss_final = np.mean(losses)
        results["acc_top1"] = np.mean(results["acc_top1"])
        results["acc_top5"] = np.mean(results["acc_top5"])
        return loss_final, results


class BaseRegressionTrainer(BaseTrainer):

    @torch.no_grad()
    def evalulate(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        device: torch.device,
    ) -> tuple[float, Dict]:
        raise NotImplementedError

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        data_loader: Iterable,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
    ):
        raise NotImplementedError
