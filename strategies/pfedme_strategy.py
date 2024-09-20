from typing import Callable, Union, OrderedDict
from .base_strategy import BaseStrategy, weighted_metrics_avg
import numpy as np
import torch

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Dict, List, Optional, Tuple
import flwr as fl
from utils import build_model, get_parameters


class pFedMeStrategy(BaseStrategy):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.config = config
        self.alpha = config["alpha"]

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        self.net = build_model(self.config["model_name"], self.config["model_kwargs"])
        ndarrays = get_parameters(self.net)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def get_parameters(self) -> NDArrays:
        """Get current global model parameters."""
        parameters: NDArrays = [
            val.cpu().numpy() for _, val in self.net.state_dict().items()
        ]
        return parameters

    # def initialize_parameters(
    #     self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     net = build_model(self.config["model_name"], self.config["model_kwargs"])
    #     ndarrays = get_parameters(net)
    #     self.stat_dict = net.state_dict()
    #     return fl.common.ndarrays_to_parameters(ndarrays)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using pFedMe aggregation."""
        # Calculate the weighted average of client models
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        npnumber_aggregated = aggregate(weights_results)
        # Calculate the weighted average of current global model and aggregated model
        # pFedMe aggregation: weighted average of current global model and aggregated model
        current_parameters = (
            self.get_parameters()
        )  # Assuming this method retrieves current global model parameters

        for current_value, aggregated_value in zip(
            current_parameters, npnumber_aggregated
        ):
            current_value = (
                self.alpha * current_value + (1 - self.alpha) * aggregated_value
            )

        parameters_aggregated = ndarrays_to_parameters(npnumber_aggregated)
        self.set_parameters(current_parameters)
        # Aggregate metrics
        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]
        metrics_aggregated = weighted_metrics_avg(metrics)
 
        return parameters_aggregated, metrics_aggregated

    def set_parameters(self, parameters: NDArrays) -> None:
        """set parameters according to the given parameters

        Args:
            parameters (NDArrays): the given parameters
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # now replace the parameters
        self.net.load_state_dict(state_dict, strict=True)
