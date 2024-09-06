import os
import sys

from loguru import logger


def jetstream_pt_available() -> bool:
    """Check if the necessary imports to use jetstream_pt are available.
    """
    try:
        # For now Jetstream Pytorch is opt-in, it can be enabled with an ENV variable.
        jetstream_pt_enabled = os.environ.get("JETSTREAM_PT", False) == "1"
        if not jetstream_pt_enabled:
            return False
        # Torch XLA should not be imported before torch_xla2 to avoid conflicts.
        if 'torch_xla2' not in sys.modules and 'torch_xla.core' in sys.modules:
            logger.warning("torch_xla2 cannot be imported after torch_xla, disabling Jetstream PyTorch support.")
            return False
        # Import torch_xla2 first!
        import torch_xla2  # noqa: F401, isort:skip

        import jetstream_pt  # noqa: F401

        return True
    except ImportError:
        return False
