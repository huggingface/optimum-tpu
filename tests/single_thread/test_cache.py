import os
from tempfile import TemporaryDirectory

import torch

from optimum.tpu import initialize_cache


def test_init_cache():
    os.environ["PJRT_DEVICE"] = "TPU"
    # This is just to make sure the model has been downloaded
    with TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        initialize_cache(cache_dir)
        assert not os.path.exists(cache_dir)

        # Do some calculation that will trigger graph generation and caching
        v1 = torch.ones((100, 200), device="xla")
        v2 = torch.ones((200, 100), device="xla")
        v3 = v1 @ v2
        # Result is printed to avoid the optimizer to remove the computation
        print(v3.max())

        assert os.path.exists(cache_dir)
        assert len(os.listdir(cache_dir)) > 0


