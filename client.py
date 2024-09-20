from data.dataset import load_dataset
# from data.data_manager import Next_stream_data
import argparse
from omegaconf import OmegaConf
from strategies.strategy_factory import strategy_factory
from clients.client_factory import client_factory
from utils import seed_everything
import flwr as fl

# running a client
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
    parser.add_argument(
        "--dry",
        type=str,
        help="perfrom test (test the functions of the clients)",
        default=False,
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    print("------------------------")
    print("training configs:", config)
    print("------------------------")
    # 2. load dataset and finish the non iid data
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

    def client_fn(cid: str):
        return client_class(
            train_set=client_data[int(cid)], test_set=None, config=config, idx=int(cid)
        )

    client = client_fn(cid=args.idx)
    if args.dry:
        params = client.get_parameters({})
        results = client.fit(params, {})
        print(
            "test fit finished: num_examples_trained:{},training_status:{}".format(
                results[1], results[2]
            )
        )
        results = client.evaluate(params, {})
        print("test evaluate finished: eval results = {}".format(results))
    else:
        fl.client.start_numpy_client(
            server_address=f"127.0.0.1:{args.port}", client=client
        )
