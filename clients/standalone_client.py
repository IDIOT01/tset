from typing import Dict
from torch.utils.data.dataset import Dataset
from .base_client import BaseClient


class StandaloneClient(BaseClient):
    def __init__(
        self, train_set: Dataset, test_set: Dataset, config: Dict, idx: int
    ) -> None:
        super().__init__(train_set, test_set, config, idx)
