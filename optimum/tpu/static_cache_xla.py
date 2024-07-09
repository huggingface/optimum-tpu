from typing import Any, Dict, Optional, Tuple

import torch
from transformers import StaticCache


class StaticCacheXla(StaticCache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        # `index_copy_(dim, index, source)` functions similarly to `tensor[index] = source`,
        # but it is used for better generality and it uses less memory on XLA.
        # For more information, refer to: https://pytorch.org/cppdocs/notes/tensor_indexing.html
        k_out.index_copy_(2, cache_position, key_states)
        v_out.index_copy_(2, cache_position, value_states)

        return k_out, v_out


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        key_cache = self.key_cache[layer_idx]
        device = key_cache.device

        # index_select(dim, index) performs the same operation as item = tensor[..., index, ...]
        # but it is used for better generality and it uses less memory on XLA.
        # For more information, refer to: https://pytorch.org/cppdocs/notes/tensor_indexing.html
        item = key_cache.index_select(0, torch.tensor(0, device=device))
        head = item.index_select(1, torch.tensor(0, device=device))

        return head.any(dim=-1).sum()
