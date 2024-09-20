from typing import Callable, Dict, Tuple, Union, Optional, List
import flwr as fl
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
from .base_strategy import BaseStrategy
import torch
from utils import build_model, get_parameters
import numpy as np


def test_rank(net_blank, toSelect: NDArrays, select_list: List[NDArrays], cos=torch.nn.CosineSimilarity(dim=0)):
    if len(select_list) == 0:
        return []
  
    # prepare the parameter
    toSelect_dict = {name: param for name, param in zip(net_blank.state_dict().keys(), toSelect)}
  
    tmp_model_para = {}
    for name, param in toSelect_dict.items():
        if "fc" in name or "class" in name:
            tmp_model_para[name] = param
  
    # prepare the rank_list
    result_simi = []
  
    # start to compute similarity for each model and toSelect
    for model in select_list:
        tmp_model_dict = {name: param for name, param in zip(net_blank.state_dict().keys(), model)}
        simi = 0
        for key in tmp_model_para:
            # 确保相应的键存在于两个字典中
            if key in tmp_model_dict:
                similar = cos(torch.from_numpy(tmp_model_para[key]), torch.from_numpy(tmp_model_dict[key])).mean().item()
                simi += similar
        simi = simi / len(tmp_model_para)
        result_simi.append(simi)
  
    return result_simi


class FedHISTStrategy(BaseStrategy):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.para_pool = [] # the pool save the parameters
        self.net_blank = build_model(self.config["model_name"], self.config["model_kwargs"])
        self.para_client = [[] for _ in range(config['num_clients'])] #store the client's parameters -- store the Parameters

    def __repr__(self) -> str:
        return "FedHist Strategy"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
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
            # standard_config = {'cid':idx}
            fit_configurations.append((client, FitIns(ndarrays_to_parameters(self.para_client[idx]) if len(self.para_client[idx])>0 else parameters , standard_config)))
        
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # 1. store the updated mfd parameters (whole model)
        self.para_client = [
            parameters_to_ndarrays(fit_res.parameters)
            for _,fit_res in results
        ]
        # 2. test the similarity for each client model
        for idx in range(len(self.para_client)):
            if len(self.para_pool) == 0:
                self.para_pool.append(self.para_client[idx])
                continue
            rank = test_rank(self.net_blank,self.para_client[idx],self.para_pool)
            if len(rank) == 0 or max(rank) < self.config['similar_K']:
                self.para_pool.append(self.para_client[idx])
                continue
            else:
                rank_id = np.argmax(rank)
                tmp = [self.para_pool[i] for i in range(len(self.para_pool)) if rank[i] >= self.config['similar_K']]

                # 首先，确保 tmp 不是空的
                if len(tmp) > 0:
                    # 初始化用于存储均值参数的列表
                    mean_params = []
                    # 获取参数的数量（假设所有模型参数数量相同）
                    num_params = len(tmp[0])
                    # 对每个参数位置
                    for i in range(num_params):
                        # 取出所有模型在这个位置的参数
                        param_list = [model[i] for model in tmp]
                        # 计算这些参数的均值（沿着第0个轴，即不同模型间）
                        param_mean = np.mean(np.array(param_list), axis=0)
                        # 将均值参数添加到列表中
                        mean_params.append(param_mean)

                    # 现在，mean_params 包含了所有参数的均值                
                    self.para_pool[rank_id][:] = mean_params
                    self.para_client[idx] = mean_params
        # 3. send to the client

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results]
        metrics_aggregated = weighted_metrics_avg(metrics)
    
        return parameters_aggregated, metrics_aggregated

    # def configure_evaluate(
    #     self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> List[Tuple[ClientProxy, EvaluateIns]]:
    #     pass

    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     pass
def weighted_metrics_avg(metrics: list[tuple[Dict[str, Scalar], int]]) -> dict:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for _, num_examples in metrics])
    # Initialize an empty dictionary to store the aggregated metrics
    aggregated_metrics = {}
    # Loop over the keys of the metrics dictionary
    if len(metrics) > 0:
        for key in metrics[0][0].keys():
            # Calculate the weighted average of the metric values from all clients
            weighted_sum = sum(
                [metric[key] * num_examples for metric, num_examples in metrics]
            )
            weighted_avg = weighted_sum / num_total_evaluation_examples
            # Store the weighted average value in the aggregated metrics dictionary
            aggregated_metrics[key] = weighted_avg
    # Return the aggregated metrics dictionary
    return aggregated_metrics
