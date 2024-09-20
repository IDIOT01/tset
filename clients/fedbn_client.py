import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import flwr as fl
from torch.utils.data.dataset import Dataset
from .base_client import BaseClient


class FedBNClient(BaseClient):
    """Similar to FlowerClient but this is used by FedBN clients."""

    def __init__(
        self, train_set: Dataset, test_set: Dataset, config: Dict, idx: int
    ) -> None:
        super().__init__(train_set, test_set, config, idx)
        # IMPORTANT NOTE:
        # For FedBN clients we need to persist the state of the BN
        # layers across rounds. In Simulation clients are statess
        # so everything not communicated to the server (as it is the
        # case as with params in BN layers of FedBN clients) is lost
        # once a client completes its training. An upcoming version of
        # Flower suports stateful clients
        save_path = Path(self.config["save_path"])
        bn_state_dir = save_path / "bn_states"
        bn_state_dir.mkdir(exist_ok=True)
        self.bn_state_pkl = bn_state_dir / f"client_{idx}.pkl"

    def _save_bn_statedict(self) -> None:
        """Save contents of state_dict related to BN layers."""
        bn_state = {
            name: val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" in name
        }

        with open(self.bn_state_pkl, "wb") as handle:
            pickle.dump(bn_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_bn_statedict(self) -> Dict[str, torch.Tensor]:
        """Load pickle with BN state_dict and return as dict."""
        with open(self.bn_state_pkl, "rb") as handle:
            data = pickle.load(handle)
        bn_stae_dict = {k: torch.from_numpy(v) for k, v in data.items()}
        return bn_stae_dict

    def get_parameters(self, config) -> NDArrays:
        """Return model parameters as a list of NumPy ndarrays w or w/o using BN.

        layers.
        """
        # First update bn_state_dir
        self._save_bn_statedict()
        # Excluding parameters of BN layers when using FedBN
        return [
            val.cpu().numpy()
            for name, val in self.model.state_dict().items()
            if "bn" not in name
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy ndarrays Exclude the bn layer if.

        available.
        """
        keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

        # Now also load from bn_state_dir
        if self.bn_state_pkl.exists():  # It won't exist in the first round
            bn_state_dict = self._load_bn_statedict()
            self.model.load_state_dict(bn_state_dict, strict=False)
