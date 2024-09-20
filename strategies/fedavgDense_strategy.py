from typing import Callable, Union, OrderedDict
from .base_strategy import BaseStrategy, weighted_metrics_avg
import numpy as np
import torch
import wandb
import torch.nn as nn
from utils import train, evaluate

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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from datetime import datetime
import csv
import os


class FedAvgDenseStrategy(BaseStrategy):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.numTest = 0
        # self.convLen = 0
        self.net = None
        self.g_e = 0
        self.public_trainset = None
        self.public_testset = None
        self.run = None
        self.traffic = 0

        # 从配置中获取 CSV 路径和文件名
        csv_path = self.config['csv_path']
        csv_name = (
            f"{self.config['csv_name']}_topk{self.config['global_model_kwargs']['top_k']}"
            f"_dirichlet_alpha{self.config['dirichlet_alpha']}_dataset{self.config['dataset']}.csv"
        )
        self.csv_file = os.path.join(csv_path, csv_name)  # 创建完整的 CSV 文件路径

        # 确保 CSV 路径存在
        os.makedirs(csv_path, exist_ok=True)

        # 存储所有记录的数据
        self.local_log = []

        # 写入 CSV 标题行的标志
        self.csv_initialized = False

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        print("initialize_parameters{}".format(self.numTest))
        self.numTest += 1
        project_name = '1'
        runs_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
        runs_name = 'cnn_moe' + '_at_' + runs_time
        temp_config = {
            'alpha': self.config['dirichlet_alpha'],
            'batch_size': self.config['batch_size'],
            'class_num': self.config['model_kwargs']['class_num'],
            'client_num': self.config['num_clients'],
            'device': self.config['device'],
            'expert_num': self.config['global_model_kwargs']['expert_num'],
            'fine_tuning_epochs': self.config['fine_tune_epoch'],
            # 'gating_hidden_dim': self.config['gating_hidden_dim'],
            'global_epochs': self.config['rounds'],
            'local_epochs': self.config['local_epoch'],
            'mlp_hidden_dim': self.config['model_kwargs']['mlp_hidden_dim'],
            'public_rate': self.config['public_rate'],
            'top_k': self.config['global_model_kwargs']['top_k'],
        }
        self.run = wandb.init(project=project_name, name=runs_name, config=temp_config)

        self.net = build_model(self.config["global_model_name"], self.config["global_model_kwargs"])
        ndarrays = get_parameters(self.net)
        self.public_trainset, self.public_testset = loadDataSet(self.config["batch_size"], self.config["public_rate"], self.config["dataset"])
        # self.convLen = len(ndarrays)
        return fl.common.ndarrays_to_parameters(ndarrays)

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
            fit_configurations.append((client, FitIns(parameters, standard_config)))

        return fit_configurations

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        results_sorted = sorted(results, key=lambda x: x[0].cid)
        
        
        def get_torch_model_size(parameters):
            total_size = 0
            for p in parameters:
                element_size = torch.tensor([], dtype=p.dtype).element_size()
                total_size += p.numel() * element_size
            return total_size / (1024 ** 2)  # 转换为 MB

        # 计算模型参数大小的辅助函数
        def get_model_size(parameters):
            total_size = sum(p.nbytes for p in parameters)  # NumPy数组的总字节数
            return total_size / (1024 ** 2)  # 转换为 MB
        
            # 计算聚合之前的参数总大小
        self.traffic += sum(
            get_model_size(parameters_to_ndarrays(fit_res.parameters))
            for _, fit_res in results_sorted
        )
        print(f"Total parameters size before aggregation: {self.traffic:.2f} MB")

        # 聚合权重
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results_sorted
        ]
        # parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        aggregate_para = aggregate(weights_results)
        # conv_weights_aggregated = ndarrays_to_parameters(aggregate_para)

        # 更新全局模型权重
        model_state_dict = self.net.state_dict().keys()
        # Ensure the length of parameters matches the cnn_state_dict
        assert len(aggregate_para) == len(model_state_dict), "Mismatch in the number of CNN parameters"

        # Create a state_dict for the CNN parti
        aggregate_para_as_ndarray = [np.array(v) if not isinstance(v, np.ndarray) else v for v in aggregate_para]
        params_dict = {k: torch.from_numpy(v) for k, v in zip(model_state_dict, aggregate_para_as_ndarray)}


        self.net.load_state_dict(params_dict, strict=True)

        # ft the global model at the server side
        self.fine_tune_server_model()

        self.traffic += get_torch_model_size(self.net.parameters())

        metrics = [(fit_res.metrics, fit_res.num_examples) for _, fit_res in results_sorted]
        metrics_aggregated = weighted_metrics_avg(metrics)

        # 将聚合后的参数转换回参数格式
        parameters_aggregated = ndarrays_to_parameters(aggregate_para)

        return parameters_aggregated, metrics_aggregated


    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        print("configure_evaluate{}".format(self.numTest))
        self.numTest += 1
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
        print("aggregate_evaluate{}".format(self.numTest))
        self.numTest += 1

        return None, {}

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        print("evaluate{}".format(self.numTest))
        self.numTest += 1

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def fine_tune_server_model(self):

        for param in self.net.conv.parameters():  # 冻结conv和expertlist
            param.requires_grad = False
        for expert in self.net.experts:
            for param in expert.parameters():
                param.requires_grad = False
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(self.net.w_gate.parameters()) + list(self.net.w_noise.parameters()),
                               lr=0.001)

        g_e = self.g_e

        fine_tuning_epochs = self.config['fine_tune_epoch']
        device = self.config['device']
        for f_e in range(fine_tuning_epochs):
            # self.net.noisy_gating = True
            # train_loss = train(model=self.net, train_loader=self.public_trainset, criterion=criterion,
            #                    optimizer=optimizer, device=device)
            # self.net.noisy_gating = False
            # loss_train, acc_train = evaluate(model=self.net, test_loader=self.public_trainset, criterion=criterion,
            #                                  device=device)
            loss, acc = evaluate(model=self.net, test_loader=self.public_testset, criterion=criterion, device=device)
            # traffic_step = int(self.traffic)

            training_round = g_e * fine_tuning_epochs + f_e
            log_data = {
                'test_loss': loss,
                'accuracy': acc,
                # 'training_loss': train_loss,
                'training_round': training_round,
                'traffic': self.traffic,
                'communication_round': self.g_e,
            }

            # 记录到 wandb
            wandb.log(log_data)

            # 将数据添加到 local_log 列表中
            self.local_log.append(log_data)
            print('*' * 80)
            print('Global Model in Round {} and Epoch {}. Loss: {}, Accuracy: {}.'.format(g_e, f_e, loss, acc))
            print(self.g_e)
            print(training_round)
            print(self.config['rounds'])
            print('*' * 80)

        self.g_e += 1
        # 结束时将所有数据写入 CSV 文件
        if self.g_e == self.config['rounds']:
            self._write_log_to_csv()

            print('Already write Log!')


        return None
    def _write_log_to_csv(self):
        """将记录的数据写入 CSV 文件"""
        if not self.csv_initialized:
            # 写入 CSV 标题行
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'test_loss', 
                    'accuracy', 
                    'training_round', 
                    'traffic', 
                    'communication_round'
                ])
            self.csv_initialized = True

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

def loadDataSet(batch_size, public_rate, datasetName):
    # 根据datasetName确定使用的预处理方式
    if datasetName == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        # 加载CIFAR-10数据集
        trainset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    elif datasetName == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 将28x28的图像调整为32x32
            transforms.ToTensor(),  # 转换为张量
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将1通道图像复制为3通道
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化
        ])
        # 加载MNIST数据集，并进行相同的处理
        trainset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {datasetName}")

    # 随机抽取公共数据集
    indices = list(range(len(trainset)))
    np.random.shuffle(indices)
    sample_size = int(len(trainset) * public_rate)
    sample_indices = indices[:sample_size]
    sample_subset = Subset(trainset, sample_indices)
    public_trainset = DataLoader(sample_subset, batch_size=batch_size, shuffle=True)
    public_testset = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return public_trainset, public_testset

def parameters_to_state_dict(parameters, model):
    state_dict = model.state_dict()
    param_names = list(state_dict.keys())
    for i, param in enumerate(parameters):
        state_dict[param_names[i]].copy_(param.data)
    return state_dict
