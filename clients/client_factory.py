from .base_client import BaseClient
from .fedbn_client import FedBNClient
from .fedhist_client import FedHISTClient
from .fedft_client import FedFTClient
from .fedftavgDense_client import FedAvgDenseClient
from .fedstream_client import FedStreamClient
from .fedprox_client import FedProxClient
from .fedRA_client import FedRAClient
from .pfedme_client import pfedmeClient
from .standalone_client import StandaloneClient
from typing import Type

CLIENT_LIST = {
    "FedAvg": BaseClient,
    "FedBN": FedBNClient,
    # add other
    "pFedMe": pfedmeClient,
    "FedHist": FedHISTClient,
    "Standalone": StandaloneClient,
    "FedFT": FedFTClient,
    "FedAvgDense": FedAvgDenseClient,
    "FedProx": FedProxClient,
    "FedRA": FedRAClient,
}


def client_factory(algorithm: str) -> Type[BaseClient]:
    """get client class according to different algorithms

    Args:
        algorithm (str): your fl algorithm

    Returns:
        type[BaseClient]: the client class
    """
    return CLIENT_LIST[algorithm]
