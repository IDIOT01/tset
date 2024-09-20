import flwr as fl
from typing import Dict, Tuple, OrderedDict

from flwr.common import Config, NDArrays, Scalar
from torch.utils.data.dataset import Dataset
from .base_client import BaseClient
from data.data_manager import *
from utils import (
    build_model,
    build_trainer,
    build_optimizer,
    build_criterion,
    get_parameters,
)

class FedHISTClient(BaseClient):
    def __init__(
        self, train_set: Dataset, test_set: Dataset, config: Dict, idx: int
    ) -> None:
        super().__init__(train_set, test_set, config, idx)
        self.mfd = copy.deepcopy(self.model) # model for data
        self.mfd_ini = copy.deepcopy(self.model) # each round using mfd_ini to initialize mfd
        # self.mft = copy.deepcopy(self.model) # model for task
        self.streamingData = None # the dataloader of this round

    def set_parameters(self, parameters: NDArrays) -> None:
        """set parameters according to the given parameters

        Args:
            parameters (NDArrays): the given parameters
        """
        # average 
        model_state = self.model.state_dict()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        for name, param in params_dict:
            if 'fc' in name or 'class' in name:
                if not isinstance(param, torch.Tensor):
                    param = torch.Tensor(param)
                model_state[name] = param

        # params_dict = zip(self.model.state_dict().keys(), parameters)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
                
        # now replace the parameters
        self.model.load_state_dict(model_state, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
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

        # step 3: get the next stream of data        
        self.train_loader, self.test_loader = Next_stream_data(
            self.train_set, self.config, self.idx
        )
        self.mfd.load_state_dict(self.mfd_ini.state_dict())
        optimizer = build_optimizer(
            self.config["optimizer"],
            self.mfd.parameters(),
            self.config["optimizer_kwargs"],
        )
        criterion = build_criterion(self.config["criterion"])
        _ = self.trainer.train(
            self.mfd,
            criterion,
            self.train_loader,
            optimizer,
            self.config["local_epoch"],
            self.config["device"],
        )
        
        # step 4: return the training results
        parameters_prime = self.get_parameters_mfd(None)
        num_examples_train = len(self.train_loader.dataset)
        train_status = {"training_loss": training_loss}


        return parameters_prime, num_examples_train, train_status

    def get_parameters_mfd(self, config: Dict) -> NDArrays:
        parameters: NDArrays = [
            val.cpu().numpy() for _, val in self.mfd.state_dict().items()
        ]
        return parameters