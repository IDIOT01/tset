from typing import Dict
import flwr as fl
from typing import Dict, Tuple

from flwr.common import Config, NDArrays, Scalar
from torch.utils.data.dataset import Dataset
from .base_client import BaseClient


class FedStreamClient(BaseClient):
    def __init__(
        self, train_set: Dataset, test_set: Dataset, config: Dict, idx: int
    ) -> None:
        super().__init__(train_set, test_set, config, idx)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        pass

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        pass

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        pass
