import flwr as fl
from typing import Dict, Tuple, OrderedDict
import torch
from flwr.common import Config, NDArrays, Scalar
from torch.utils.data.dataset import Dataset
import torch.nn as nn
from utils import (
    build_model,
    build_trainer,
    build_optimizer,
    build_criterion,
    get_parameters,
)

from data.data_manager import client_load_data
from scipy.spatial.distance import cosine
import numpy as np


class FedFTClient(fl.client.NumPyClient):
    def __init__(
            self,
            train_set: Dataset,
            test_set: Dataset,
            config: Dict,
            idx: int,
    ) -> None:
        super().__init__()
        # save arguments to member variables
        self.train_set = train_set
        self.test_set = test_set
        self.config = config
        self.idx = idx
        self.model: nn.Module = build_model(
            self.config["model_name"], self.config["model_kwargs"]
        )
        self.trainer = build_trainer(self.config["task"], self.config["algorithm"])

        self.train_loader, self.test_loader = client_load_data(
            self.train_set, self.config, self.idx
        )
        self.mlpLen = len(self.model.mlp.state_dict().keys())
        # print(self.mlpLen)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set parameters according to the given parameters.

        Args:
            parameters (NDArrays): The given parameters.
        """
        # Get the state_dict keys for the CNN part of the client model
        cnn_state_dict = self.model.conv.state_dict().keys()

        # Update the CNN parameters
        conv_params_dict = {k: torch.from_numpy(v) for k, v in zip(cnn_state_dict, parameters[:len(cnn_state_dict)])}

        # Load the CNN parameters into the CNN part of the model
        self.model.conv.load_state_dict(conv_params_dict, strict=True)

        # Calculate the starting index for the expert parameters
        expert_start_idx = len(cnn_state_dict)

        # Determine the number of experts
        num_experts = (len(parameters) - expert_start_idx) // self.mlpLen

        # If there are no expert parameters, do not update the MLP part
        if num_experts == 0:
            print("No expert models returned from the server. Skipping MLP update.")
            return

        # Flatten the local expert model's parameters for comparison
        local_expert_weights_flattened = np.concatenate(
            [layer.detach().cpu().flatten().numpy() for layer in self.model.mlp.parameters()]
        )

        # Initialize variables to track the closest expert
        closest_expert_idx = None
        closest_expert_distance = float('inf')

        # Iterate through the expert parameters returned from the server
        for i in range(num_experts):
            # Extract and flatten the current expert's parameters from the server
            server_expert_params = parameters[
                                   expert_start_idx + i * self.mlpLen:expert_start_idx + (i + 1) * self.mlpLen]
            server_expert_weights_flattened = np.concatenate([param.flatten() for param in server_expert_params])

            # Compute cosine distance between the local and server expert models
            distance = cosine(local_expert_weights_flattened, server_expert_weights_flattened)

            # Update closest expert index if a closer one is found
            if distance < closest_expert_distance:
                closest_expert_distance = distance
                closest_expert_idx = i

        # Use the parameters of the closest expert to update the local expert model
        closest_expert_params = parameters[expert_start_idx + closest_expert_idx * self.mlpLen:
                                           expert_start_idx + (closest_expert_idx + 1) * self.mlpLen]
        expert_state_dict = self.model.mlp.state_dict().keys()

        # Create a state_dict for the expert part
        expert_params_dict = {k: torch.from_numpy(v) for k, v in zip(expert_state_dict, closest_expert_params)}

        # Load the parameters into the expert model
        self.model.mlp.load_state_dict(expert_params_dict, strict=True)
        print('success update client MLP model')

    def fit(
            self, parameters: NDArrays, training_ins: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """train the local models accoring to instructions

        Args:
            parameters (NDArrays): the given parameters
            training_ins (Dict[str, Scalar]): training instructions

        Returns:
            Tuple[NDArrays, int, Dict[str, Scalar]]: the trained parameters, the number of sample trained, the training statistics.
        """
        # step 1: receive the parameters from server
        self.set_parameters(parameters)

        # step 2: train the model according to training instructions
        optimizer = build_optimizer(
            self.config["optimizer"],
            self.model.parameters(),
            self.config["optimizer_kwargs"],
        )
        criterion = build_criterion(self.config["criterion"])
        training_loss = self.trainer.train(
            self.model,
            criterion,
            self.train_loader,
            optimizer,
            self.config["local_epoch"],
            self.config["device"],
        )
        # step 3: return the training results
        parameters_prime = self.get_parameters(None)
        num_examples_train = len(self.train_loader.dataset)
        train_status = {"training_loss": training_loss}

        return parameters_prime, num_examples_train, train_status

    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        # step 1: receive the parameters from server
        # self.set_parameters(parameters)
        #
        # # step 2: evaluate the model according to the instructions
        # criterion = build_criterion(self.config["criterion"])
        # eval_loss, eval_status = self.trainer.evalulate(
        #     self.model, criterion, self.test_loader, self.config["device"]
        # )

        # step 3: return the test loss and other statistics
        # return eval_loss, len(self.test_loader.dataset), eval_status
        return 0.0, 0, {}

    def get_parameters(self, config: Dict) -> NDArrays:
        parameters: NDArrays = [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]
        return parameters
# 这里考虑MLP和CNN两个分开return，方便global更新聚合
