import os


def jetstream_pt_available() -> bool:
    """Check if the necessary imports to use jetstream_pt are available.
    """
    try:
        # Jetstream Pytorch is enabled by default, it can be disabled with an ENV variable.
        jetstream_pt_disabled = os.environ.get("JETSTREAM_PT_DISABLE", False) == "1"
        if jetstream_pt_disabled:
            return False
        # Import torch_xla2 first!
        import torch_xla2  # noqa: F401, isort:skip

        import jetstream_pt  # noqa: F401

        return True
    except ImportError:
        return False
