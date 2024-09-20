# main entry of the program
from omegaconf import OmegaConf
from clients.client_factory import client_factory
from clients.base_client import BaseClient
import argparse
from data.dataset import load_dataset
from strategies.strategy_factory import strategy_factory
from utils import seed_everything
import flwr as fl
import os
import time
import wandb

if __name__ == "__main__":
    # 1. load configs
    parser = argparse.ArgumentParser(description="Streaming FL configs")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Configuration file"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print("------------------------")
    print("training configs:", config)
    print("------------------------")

    # 2. load dataset
    client_data = load_dataset(
        config["dataset"],
        config["data_root"],
        config["num_clients"],
        config["iid_degree"],
    )
    # 3. maunally set seed for reproduction
    seed_everything(config["seed"])

    # 4. define clien_fn
    client_class = client_factory(config["algorithm"])

    def client_fn(cid: str) -> BaseClient:
        return client_class(
            train_set=client_data[int(cid)], test_set=None, config=config, idx=int(cid)
        )

    # 5. log settings
    log_path = os.path.join(config["log_dir"], config["algorithm"])
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(
        log_path,
        time.strftime(
            "%Y-%m-%d_%H_%M_%S",
        )
        + ".log",
    )
    fl.common.logger.configure(
        identifier=config["dataset"] + "_" + config["algorithm"], filename=log_file
    )
    # wandb.init(
    #     project="StreamingFL",
    #     name=f"{config["dataset"]}_{config["exp_name"]}",
    #     # Track hyperparameters and run metadata
    #     config=config,
    # )
    # 6. fun simulation
    strategy_class = strategy_factory(config["algorithm"])
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=config["rounds"]),
        strategy=strategy_class(config),  # <-- pass the new strategy here
        client_resources=config["client_resources"],
    )
