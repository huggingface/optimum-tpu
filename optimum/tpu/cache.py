from pathlib import Path
from typing import Union

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def initialize_cache(path: Union[str, Path] = "~/.cache/optimum-tpu/"):
    """Initialize the cache for the XLA runtime.

    Note that this will only initialize the cache on the master ordinal.

    Args:
        path (`str`, defaults to `~/.cache/optimum-tpu/`):
            The path to the cache directory.
    """
    # Resolve tilde in the path
    path = Path(path).expanduser()
    # It will be readonly only if the ordinal is not 0, i.e. not the master
    readonly = xm.get_ordinal() != 0
    xr.initialize_cache(str(path), readonly=readonly)
