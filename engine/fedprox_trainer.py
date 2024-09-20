from .base_trainer import BaseCVTrainer
from typing import Dict, Iterable
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sys

from flwr.common import Config, NDArrays, Scalar


class FedProxCVTrainer(BaseCVTrainer):
    def __init__(self) -> None:
        super().__init__()

    def train(
        self,
        model: nn.Module,
        global_params: NDArrays,
        criterion: nn.Module,
        data_loader: Iterable,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
        proximal_mu: float,
    ) -> float:
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
                    # calculate the proximal term: IMPORTANT OPERATIONS!!

                    proximal_term = torch.tensor(0, dtype=torch.float64, device=device)
                    for (param_name, local_weights), global_weights in zip(
                        model.state_dict().items(), global_params
                    ):
                        # Debug
                        # print(
                        #     "{}->{},{}".format(
                        #         param_name, local_weights.dtype, local_weights.shape
                        #     )
                        # )
                        # SKIP num_batches_tracked
                        if "num_batches_tracked" not in param_name:
                            proximal_term += (
                                (
                                    local_weights
                                    - torch.from_numpy(global_weights).to(
                                        local_weights.device
                                    )
                                )
                                .type(torch.float64)
                                .norm(2)
                            )
                    loss += proximal_mu * proximal_term
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
