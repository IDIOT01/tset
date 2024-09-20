# the server
import argparse
from omegaconf import OmegaConf
from strategies.strategy_factory import strategy_factory
import os, time
import flwr as fl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower client")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config_path of the client, should be consistent across all clients.",
        default="configs/template.yaml",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        required=False,
        help="The server port",
        default="8081",
    )
    parser.add_argument(
        "--idx",
        type=str,
        required=False,
        help="index of the client",
        default="0",
    )
    args = parser.parse_args()
    # 1. load config
    config = OmegaConf.load(args.config)
    print("------------------------")
    print("training configs:", config)
    print("------------------------")
    # 2. build strategy
    strategy_class = strategy_factory(config["algorithm"])
    # 3. logging settings
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
    # 4.start the server
    fl.server.start_server(
        server_address=f"127.0.0.1:{args.port}",
        config=fl.server.ServerConfig(num_rounds=config["rounds"]),
        strategy=strategy_class(config),
    )
