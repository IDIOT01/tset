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


class StandaloneStrategy(BaseStrategy):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.weights = []

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size = int(self.config["num_clients"] * self.config["clients_fraction"])
        # optional wait for function
        client_manager.wait_for(sample_size)

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=sample_size
        )

        # Create custom configs
        standard_config = {}
        fit_configurations = []
        for idx, client in enumerate(clients):
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        self.weights[idx] if len(self.weights) != 0 else parameters,
                        standard_config,
                    ),
                )
            )

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        # IMPORTANT OPERATION: Store the results for the next iteration
        self.weights = [
            ndarrays_to_parameters(weights) for weights, _ in weights_results
        ]

        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]
        metrics_aggregated = weighted_metrics_avg(metrics)
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size = int(self.config["num_clients"] * self.config["clients_fraction"])
        # optional wait for function
        client_manager.wait_for(sample_size)

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=sample_size
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]

        metrics_aggregated = weighted_metrics_avg(metrics)
        return loss_aggregated, metrics_aggregated
