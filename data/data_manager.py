# preprocess, load and change the distribution of the data
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
import copy
import random
from typing import List
from collections import Counter


def dirichlet_data_partition(
        dataset_ini: Dataset, num_clients=10, alpha=0.5
) -> List[Dataset]:
    """split dataset into num_clients clusters using dirichle distribution

    Args:
        dataset_ini (Dataset): the initial dataset
        num_clients (int, optional): Defaults to 10.
        alpha (float, optional): dirichlet distribution alpha. Defaults to 0.5.

    Returns:
        list[Dataset]: the dataset of every client
    """

    # 1. extract labels of the dataset
    if hasattr(dataset_ini, "targets"):
        labels = dataset_ini.targets
    elif hasattr(dataset_ini, "labels"):
        labels = dataset_ini.labels
    else:
        print("No targets attribute, try tensors[1]")
        labels = dataset_ini.tensors[1]
        # raise ValueError("Dataset must have 'targets' or 'labels' attribute")

    labels = np.array(labels)
    num_classes = len(np.unique(labels))

    # 使用Dirichlet分布生成每个类在每个客户端的比例
    distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    # 确保每个客户端至少有一个样本
    while (distribution.min(axis=1) == 0).any():
        distribution = np.random.dirichlet([alpha] * num_clients, num_classes)

    # 2. assign the data samples to different client
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = distribution[k]
        # 计算每个类别按比例分配给每个客户端的索引
        idx_k_split = np.split(
            idx_k, np.cumsum([int(p * len(idx_k)) for p in proportions[:-1]])
        )
        for client in range(num_clients):
            client_indices[client].extend(idx_k_split[client].tolist())
    # 3. return subset of each client
    return [Subset(dataset_ini, indices=idx) for idx in client_indices]


def load_data(train_dataset, config, client_id):
    """Load data for a specific client by splitting the dataset using Dirichlet distribution.

    Args:
        train_dataset (Dataset): The initial training dataset to be partitioned.
        config (Dict): Configuration dictionary containing parameters for data loading.
        client_id (int): The client ID for which the data is to be loaded.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and testing DataLoaders for the client.
    """
    # Assuming the dataset is already partitioned using Dirichlet distribution
    client_datasets = dirichlet_data_partition(train_dataset, num_clients=config["num_clients"],
                                               alpha=config["dirichlet_alpha"])
    train_dataset_client = client_datasets[client_id]

    # Split the client dataset into train and test
    data_size = len(train_dataset_client)
    test_size = int((1 - config['train_fraction']) * data_size)
    train_size = data_size - test_size
    train_dataset_client, test_dataset_client = torch.utils.data.random_split(train_dataset_client,
                                                                              [train_size, test_size])

    return DataLoader(
        train_dataset_client,
        batch_size=config["batch_size"],
        shuffle=True,
    ), DataLoader(
        test_dataset_client,
        batch_size=config["batch_size"],
        shuffle=False,
    )


def client_load_data(train_dataset, config, client_id):
    """Load data for a specific client by splitting the dataset using Dirichlet distribution.

    Args:
        train_dataset (Dataset): The initial training dataset to be partitioned.
        config (Dict): Configuration dictionary containing parameters for data loading.
        client_id (int): The client ID for which the data is to be loaded.

    Returns:
        Tuple[DataLoader, DataLoader]: The training and testing DataLoaders for the client.
    """
    # Assuming the dataset is already partitioned using Dirichlet distribution
    # client_datasets = dirichlet_data_partition(train_dataset, num_clients=config["num_clients"],
    #                                            alpha=config["dirichlet_alpha"])
    # train_dataset_client = client_datasets[client_id]
    train_dataset_client = train_dataset
    # Split the client dataset into train and test
    data_size = len(train_dataset_client)
    test_size = int((1 - config['train_fraction']) * data_size)
    train_size = data_size - test_size
    train_dataset_client, test_dataset_client = torch.utils.data.random_split(train_dataset_client,
                                                                              [train_size, test_size])

    return DataLoader(
        train_dataset_client,
        batch_size=config["batch_size"],
        shuffle=True,
    ), DataLoader(
        test_dataset_client,
        batch_size=config["batch_size"],
        shuffle=False,
    )