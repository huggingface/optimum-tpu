import copy
import logging
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
from jetstream.engine import sampling_utils
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerationMode

from .logits_process import FusedLogitsWarper


logger = logging.getLogger(__name__)


class TokenSelector:
    """Implements the token selection logic corresponding to a generation configuration.

    This class combines and uses the logits processors and stopping criteria implemented in
    the transformers library.

    The algorithm to select these objects is heavily inspired by the transformers `GenerationMixin.generate()`
    method, but the actual token selection methods are specific, and partially adapted from Jetstream/Pytorch sampling
    implementation.

    The reason why this class does not inherit from `GenerationMixin` is because it does not
    include the code to produce the tokens logits.
    Separating the production of the tokens logits from the tokens selection allows this class
    to be used with different generation paradigms, either synchronously using a single `TokenSelector` in
    `GenerationMixin.generate()` or asynchronously using multiple `TokenSelector` inside an inference endpoint.

    The constructor of this class should not be called directly: instances should be obtained by
    calling `TokenSelector.create()`.
    """

    def __init__(
        self,
        mode: GenerationMode,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        eos_token_ids: Union[int, List[int]],
        pad_token_id: int,
        logits_warper: Optional[LogitsProcessorList] = None,
        seed: Optional[int] = 0,
    ):
        self.mode = mode
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.eos_token_ids = eos_token_ids
        self.pad_token_id = pad_token_id
        self.logits_warper = logits_warper
        self.key = jax.random.PRNGKey(seed)

    @classmethod
    def create(
        cls,
        input_ids: jnp.ndarray,
        generation_config: GenerationConfig,
        model: GenerationMixin,
        max_seq_length: int,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        seed: Optional[int] = 0,
    ) -> "TokenSelector":
        r"""Creates the `TokenSelector` for a specific generation configuration.

        Args:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to parametrize the token selection.
            model (`~transformers.generation.GenerationMixin`):
                The model provides the internal helpers allowing to select the logits processors and stopping criterias.
            max_seq_length (`int`):
                The maximum number of input + generated tokens for this model. It depends on the model compilation parameters.
            stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.
            seed(`Optional[int]`):
                The optional seed for sampling. Defaults to zero.
        Return:
            The `TokenSelector` instance.
        """
        generation_config.validate()
        generation_config = copy.deepcopy(generation_config)

        unsupported_generation_flags = [
            "output_attentions",
            "output_hidden_states",
            "output_scores",
            "return_dict_in_generate",
        ]
        for flag in unsupported_generation_flags:
            if getattr(generation_config, flag, False):
                raise ValueError("{flag} is not supported for generation.")

        if generation_config.max_new_tokens is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
            generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[-1]

        min_length = generation_config.min_length
        if min_length > max_seq_length:
            raise ValueError(
                f"The minimum generation length ({min_length}) exceeds the model maximum sequence length ({max_seq_length})"
            )
        max_length = generation_config.max_length
        if max_length > max_seq_length:
            logger.warning(
                f"Adjusting the maximum generation length ({max_length}) to the model maximum sequence length ({max_seq_length})"
            )
            generation_config.max_length = max_seq_length

        # Instantiate transformers library processors and criterias
        logits_processor = model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
        )
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria = model._get_stopping_criteria(generation_config, stopping_criteria=stopping_criteria)

        # This is not supposed to happen for any of the models we support
        eos_token_id = generation_config.eos_token_id
        assert eos_token_id is not None
        # The generation requires special tokens
        eos_token_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
        if generation_config.pad_token_id is None:
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_ids[0]} for open-end generation.")
            generation_config.pad_token_id = eos_token_ids[0]

        generation_mode = generation_config.get_generation_mode()
        if generation_mode not in [GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE]:
            raise ValueError("Unsupported generation mode")

        logits_warper = None
        if generation_mode == GenerationMode.SAMPLE:
            logits_warper = FusedLogitsWarper.from_config(generation_config)

        return cls(
            mode=generation_mode,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            eos_token_ids=eos_token_ids,
            pad_token_id=generation_config.pad_token_id,
            seed=seed,
        )

    def select(self, input_ids: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
        """Select the next tokens from the candidate logits.

        Args:
            input_ids (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `jnp.ndarray`: A `jnp.ndarray` containing the selected tokens.
        """
        scores = self.logits_processor(input_ids, logits)
        if self.mode == GenerationMode.SAMPLE:
            return self._sample(scores)
        else:
            return jnp.argmax(scores, axis=-1)

    def _sample(self, scores: jnp.ndarray) -> jnp.ndarray:
        do_top_k = self.logits_warper.top_k > 0 and self.logits_warper.top_k < scores.shape[-1]
        do_top_p = self.logits_warper.top_p < 1.0 and self.logits_warper.top_p > 0.0

        if do_top_k:
            return sampling_utils.sample_topk_logits(
                scores,
                self.logits_warper.top_k,
                self.logits_warper.temperature,
                self.key,
            )
        elif do_top_p:
            return sampling_utils.sample_nucleus_topp_logits(
                scores,
                self.logits_warper.top_p,
                self.logits_warper.temperature,
                self.key,
            )

        return jax.random.categorical(self.key, scores / self.logits_warper.temperature)