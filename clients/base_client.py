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

from data.data_manager import load_data


class BaseClient(fl.client.NumPyClient):
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

        self.train_loader, self.test_loader = load_data(
            self.train_set, self.config, self.idx
        )

    def set_parameters(self, parameters: NDArrays) -> None:
        """set parameters according to the given parameters

        Args:
            parameters (NDArrays): the given parameters
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict[{k: torch.from_numpy(v) for k, v in params_dict}]
        # now replace the parameters
        self.model.load_state_dict(state_dict, strict=True)

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
        self.set_parameters(parameters)

        # step 2: evaluate the model according to the instructions
        criterion = build_criterion(self.config["criterion"])
        eval_loss, eval_status = self.trainer.evalulate(
            self.model, criterion, self.test_loader, self.config["device"]
        )

        # step 3: return the test loss and other statistics
        return eval_loss, len(self.test_loader.dataset), eval_status

    def get_parameters(self, config: Dict) -> NDArrays:
        parameters: NDArrays = [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]
        return parameters
