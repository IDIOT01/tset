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


class FedRAClient(fl.client.NumPyClient):
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
        # 负责conv层的字典
        self.client_conv_layer_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 1, 5: 2, 6: 3, 7: 4, 8: 1, 9: 2, 10: 3,
                                      11: 4, 12: 1, 13: 2, 14: 3, 15: 4, 16: 1, 17: 2, 18: 3, 19: 4}
        # 定义client到expert的映射
        self.client_expert_map = {
            0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9],
            5: [0, 2], 6: [1, 3], 7: [4, 6], 8: [5, 7], 9: [8, 0],
            10: [1, 2], 11: [3, 4], 12: [5, 6], 13: [7, 8], 14: [9, 0],
            15: [1, 4], 16: [2, 5], 17: [3, 6], 18: [7, 9], 19: [8, 0]
        }
        # self.client_conv_layer_map = {0: 1, 1: 2, 2: 3, 3: 4}
        self.conv_idx = self.client_conv_layer_map[self.idx]
        # 获取当前客户端负责的expert编号
        self.client_expert_idx = self.client_expert_map[self.idx]

    def set_parameters(self, parameters: NDArrays) -> None:
        """set parameters according to the given parameters

        Args:
            parameters (NDArrays): the given parameters
        """
        # Get the state_dict keys for the CNN part of the client model
        net_state_dict = self.model.state_dict().keys()

        # Ensure the length of parameters matches the cnn_state_dict
        assert len(parameters) == len(net_state_dict), "Mismatch in the number of CNN parameters"

        # Create a state_dict for the CNN part
        params_dict = {k: torch.from_numpy(v) for k, v in zip(net_state_dict, parameters)}

        # Load the CNN parameters into the CNN part of the model
        self.model.load_state_dict(params_dict, strict=True)

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

        # 冻结除负责的expert外的所有expert参数
        for expert_idx, expert in enumerate(self.model.experts):  # 假设 experts 是模型中的expert部分
            if expert_idx not in self.client_expert_idx:
                for param in expert.parameters():
                    param.requires_grad = False  # 冻结该expert的参数
            else:
                for param in expert.parameters():
                    param.requires_grad = True  # 解冻该expert的参数

        # 参数冻结
        for name, param in self.model.conv.named_parameters():
            if 'conv' in name or 'bn' in name:
                layer_num = int(name.split('.')[0][-1])
                if layer_num != self.conv_idx:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

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
        return_message = {"training_loss": training_loss, "idx": self.idx}

        return parameters_prime, num_examples_train, return_message

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
