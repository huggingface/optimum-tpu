import copy
import logging
import os
import time
from enum import Enum
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch_xla2
from jetstream.engine.token_utils import DEFAULT_PREFILL_BUCKETS, pad_tokens, take_nearest_length
from jetstream_pt.engine import PyTorchEngine
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.generation import GenerationConfig

from ..generator_base import Generator
from ..pb.generate_pb2 import (
    Batch,
    CachedBatch,
    FinishReason,
    GeneratedText,
    Generation,
    InfoResponse,
    NextTokenChooserParameters,
    Request,
    StoppingCriteriaParameters,
    Tokens,
)
from .engine_loader import create_engine
from .token_selector import TokenSelector


# Disable optimum-tpu warnings as it seems to block the server after a while
optimum_logger = logging.getLogger("optimum.tpu")
optimum_logger.setLevel("CRITICAL")


class Slot:
    """Represents a slot in a static batch"""

    class State(Enum):
        EMPTY = 0
        READY = 1

    def __init__(self, id: int, tokenizer: PreTrainedTokenizerBase):
        self._id = id
        self._tokenizer = tokenizer
        self.clear()

    def clear(self):
        """Clear the slot and mark it as available."""
        self._state = Slot.State.EMPTY
        self._batch_id = None
        self._request_id = None
        self._generation_config = None
        self._tokens = []
        self._selector = None
        self._generated_tokens = 0
        self._next_text_token_start = 0
        self._next_text_token_end = 0
        self._generated_text = ""
        self._next_text = ""
        self._truncate = 0
        self._seed = 0

    @property
    def id(self) -> int:
        return self._id

    @property
    def state(self) -> "Slot.State":
        return self._state

    @property
    def batch_id(self) -> int:
        return self._batch_id

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def generation_config(self) -> GenerationConfig:
        return self._generation_config

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens

    @property
    def truncate(self) -> int:
        return self._truncate

    @property
    def tokens(self) -> jax.Array:
        return self._tokens

    def assign(self, batch_id: int, request: Request, generation_config: GenerationConfig):
        """Assign a request to a slot.

        Args:
            batch_id (`int`): The id of the batch containing the request.
            request (`Request`):
                The request to be assigned. Contains the inputs and tokens selection parameters.
            generation_config (`transformers.GenerationConfig`):
                The base generation config (might be modified by the request generation parameters).
        """
        self._state = Slot.State.READY
        self._batch_id = batch_id
        self._request_id = request.id
        self._inputs = request.inputs
        self._generation_config = copy.deepcopy(generation_config)
        # Update generation config with token chooser parameters
        self._generation_config.temperature = request.parameters.temperature
        self._generation_config.top_k = request.parameters.top_k
        self._generation_config.top_p = request.parameters.top_p
        self._generation_config.typical_p = request.parameters.typical_p
        self._generation_config.do_sample = request.parameters.do_sample
        self._generation_config.repetition_penalty = request.parameters.repetition_penalty
        self._truncate = request.truncate
        self._seed = request.parameters.seed
        # TODO: watermark
        self._generation_config.max_new_tokens = request.stopping_parameters.max_new_tokens
        self._max_new_tokens = self._generation_config.max_new_tokens
        # TODO: stop_sequences, ignore_eos_token

    def reset(self, input_ids: jax.Array, selector: TokenSelector):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`jax.Array`):
                The new input_ids to use to generate the next token.
            selector: (`TokenSelector`):
                An object implementing the updated token selection logic.
        """
        self._tokens = input_ids
        self._next_text_token_start = 0
        self._next_text_token_end = self._tokens.shape[-1]
        self._next_text = ""
        self._selector = selector

    def _decode_next_tokens(
        self,
    ) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        tokens = self._tokens
        # We need to include the tokens that produced the last text to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        new_text = self._tokenizer.decode(self._tokens[self._next_text_token_start :], skip_special_tokens=False)
        if new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            return ""

        # Compare the generated text with the one using only the tokens producing the last one
        last_text = self._tokenizer.decode(
            tokens[self._next_text_token_start : self._next_text_token_end],
            skip_special_tokens=False,
        )
        if len(new_text) == len(last_text):
            # Nothing new was actually generated
            return ""
        # Return the decoded text and store its token offsets
        self._next_text_token_start = self._next_text_token_end
        self._next_text_token_end = tokens.shape[-1]
        return new_text[len(last_text) :]

    def append(self, next_token: int) -> str:
        """Append a new generated token to this slot

        The new token is added to the list of generated tokens, which impacts
        directly the generated_text and stopped property.

        The new token is however not added immediately to the slot inputs: it will
        be added later on when it has effectively been used to produce the next token.

        Args:
            next_token (`int`):
                The newly generated token.

        Return:
            The corresponding decoded text (if any).
        """
        self._tokens = jnp.concat([self._tokens, jnp.array([next_token])])
        self._generated_tokens += 1
        next_text = self._decode_next_tokens()
        # Now that a new token has been generated, we can append the previous one to the generated text
        self._generated_text += self._next_text
        self._next_text = next_text
        return next_text

    def select(self, logits: jnp.ndarray) -> int:
        """Select the next token from the candidate logits.

        Args:
            logits (`jnp.ndarray` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            int: A scalar of the selected token.
        """
        if len(logits.shape) == 1:
            logits = logits.reshape(1, -1)
        return self._selector.select(self._tokens, logits)[0]

    @property
    def stopped(self) -> bool:
        # unsqueeze tokens to avoid problems with stopping criteria
        tokens = torch_xla2.tensor.j2t(self._tokens).unsqueeze(0)
        return bool(torch.all(self._selector.stopping_criteria(tokens, None)))

    @property
    def generated_text(self) -> str:
        return self._generated_text + self._next_text

    @property
    def next_token(self) -> int:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    @property
    def empty(self) -> bool:
        return len(self._tokens) == 0

    @property
    def seed(self) -> int:
        return self._seed


class PrefillSlot:
    def __init__(self):
        self._curslot = None

    def set(self, slot: Slot):
        self._curslot = slot

    def select(self, logits: jnp.ndarray) -> int:
        return self._curslot.select(logits)

class TpuGeneratorJetStream(Generator):
    """A Generator for models running on TPU, single threaded."""

    def __init__(
        self,
        engine: PyTorchEngine,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.engine = engine
        logger.debug("Loading params (i.e. weights) on engine")
        self.params = engine.load_params()
        logger.debug("Weights loaded")
        logger.debug("Initializing decode state")
        self.decode_state = engine.init_decode_state()
        logger.debug("Decode state initialized")

        # Note: Jetstream/Pytorch requires padding to be done with 0 (at least when not specified)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer
        self.special_tokens = self.tokenizer.all_special_ids
        # Slots number is static, it cannot grow over the size of the batch
        self.slots = [Slot(i, tokenizer) for i in range(self.model.config.batch_size)]
        self.batch_id = 0
        # Note: this index will _never_ be decremented, and that's fine.
        self.slot_index = 0
        self.prefill_slot = PrefillSlot()

    @property
    def info(self) -> InfoResponse:
        """Returns the expected InfoResponse."""
        dtype = self.engine.default_dtype.dtype
        # NOTE: the device type reported is "meta", even if it's a TPU
        return InfoResponse(
            requires_padding=True,
            dtype=str(dtype),
            device_type=self.engine.env.device,
        )

    def _create_dummy_request(self, max_tokens: int) -> Batch:
        """Create a dummy request for warmup."""
        # Generate a random input with slightly more tokens than requested, because special tokens are going to be
        # skipped.
        MARGIN = 10
        input_tokens = np.random.randint(self.model.config.vocab_size, size=(1, max_tokens + MARGIN), dtype=np.int64)
        text = self.tokenizer.decode(input_tokens[0], skip_special_tokens=True)
        # These are just dummy params to allow Request creation
        parameters = NextTokenChooserParameters(
            temperature=1.0,
            top_k=None,
            top_p=None,
            do_sample=False,
            seed=None,
            repetition_penalty=1.0,
            typical_p=1.0,
        )
        stopping_parameters = StoppingCriteriaParameters(max_new_tokens=20, ignore_eos_token=True)
        dummy_request = Request(
            id=0,
            inputs=text,
            truncate=max_tokens,
            parameters=parameters,
            stopping_parameters=stopping_parameters,
        )
        return dummy_request

    def warmup(self, batch: Batch) -> int:
        """Verify if the hardware can support the target load.

        Args:
            batch (`Batch`):
                A batch corresponding to the maximum number of concurrent requests.

        Return:
            The maximum number of tokens the model supports.
        """
        logger.debug("Warming up the model")
        start = time.time()
        # Just check that the warmup request parameters match the model capacity
        batch_size = self.engine.env.batch_size
        if len(batch.requests) > batch_size:
            raise ValueError(
                f"Inconsistent server configuration: please make sure max-prefill-tokens does not exceed {batch_size} x max-input-length."
            )

        # Counter-intuitively, now we ignore the input batch. Instead, we create dummy batches to cover all possible
        # batch sizes and sequence lengths.
        seq_len = self.model.config.sequence_length
        if os.environ.get("SKIP_WARMUP", "0") == "1":
            logger.debug("Skipping warmup")
            return batch_size * seq_len
        bucket_seq_len = take_nearest_length(DEFAULT_PREFILL_BUCKETS, self.engine.max_prefill_length)
        decode_done = False
        for l in reversed(DEFAULT_PREFILL_BUCKETS):
            # Skip all the unsupported lengths
            if l > bucket_seq_len:
                continue
            # create a dummy request with the current sequence length -1 (so it gets padded up to l)
            dummy_request = self._create_dummy_request(l - 1)
            # We define few max_new_tokens to request at least one (by prefill) and another by decode.
            MAX_NEW_TOKENS = 10
            dummy_request.stopping_parameters.max_new_tokens = MAX_NEW_TOKENS
            warmup_batch = Batch(id=0,
                                    requests=[dummy_request],
                                    size=1,
                                    max_tokens=batch.max_tokens)
            logger.debug(f"Warmup for requests, len {l} seq_len {seq_len}")
            _generations, next_batch = self.prefill(warmup_batch)
            if next_batch is not None:
                self.decode([next_batch])
                decode_done = True
            self.clear()
        if not decode_done:
            logger.debug("No decode done during warmup")

        elapsed = time.time() - start
        logger.debug(f"Warmup done, took {elapsed:.2f}s")
        seq_len = self.engine.env.seq_len
        return batch_size * seq_len

    def _get_slot(self):
        """Get the next available slot."""
        for slot in self.slots:
            if slot.state == Slot.State.EMPTY:
                return slot
        # if we reach this point, all slots were used - this should not happen
        raise ValueError("All slots are used, but we should have stopped earlier")

    @property
    def model(self):
        return self.engine.pt_model

    def _token_encode(self, text: str, max_length: int) -> Tuple[jnp.ndarray, int]:
        """Tokenize the input text and return the corresponding input_ids and true_length.

        Args:
            text (`str`):
                The input text to tokenize.
            max_length (`int`):
                The maximum length of the input_ids (typically from request)
        """
        if max_length == 0:
            max_length = self.model.config.sequence_length
        # Remove one to max_length because BOS is going to be added when padding
        max_length -= 1
        input_ids = self.tokenizer.encode(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        # max_prefill_length must be a power of 2
        max_prefill_length = take_nearest_length(DEFAULT_PREFILL_BUCKETS, self.model.config.sequence_length)
        tokens, true_length = pad_tokens(input_ids[0],
                                         self.tokenizer.bos_token_id,
                                         self.tokenizer.pad_token_id,
                                         is_bos=True,
                                         max_prefill_length=max_prefill_length,
                                         jax_padding=True,
                                         )
        return tokens, true_length

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill new requests.

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """

        active_slots = [slot for slot in self.slots if slot.state == Slot.State.READY]
        len_active_slots = len(active_slots)

        len_requests = len(batch.requests)
        model_batch_size = self.model.config.batch_size
        if model_batch_size is not None and model_batch_size < len_active_slots + len_requests:
            # If raising an error here wouldn't crash the server, we could raise a ValueError
            error = ValueError(
                f"Cannot prefill {len_requests} new request(s)."
                f" Maximum batch size supported is: {model_batch_size}."
            )
            # but since it's not possible, we just log the error and return an empty generation
            logger.error(error)
            return [], None
        # Assign each request to an empty slot
        logger.debug(f"Prefilling {len_requests} new request(s) adding to {len_active_slots} active slot(s)")
        generations = []
        prefilled_active_slots = []
        for request in batch.requests:
            # Dynamically create a new slot for each request
            slot = self._get_slot()
            self.prefill_slot.set(slot)
            self.slot_index += 1
            slot.assign(self.batch_id, request, self.model.generation_config)
            logger.debug(f"Request {slot.request_id} assigned to slot {slot.id}")

            # Tokenize the inputs
            input_ids, true_lengths = self._token_encode(request.inputs, slot.truncate)
            truncated_input_ids = input_ids[:true_lengths]
            selector = TokenSelector.create(
                truncated_input_ids,
                slot.generation_config,
                self.model,
                self.model.config.sequence_length,
                seed=slot.seed,
            )
            slot.reset(truncated_input_ids, selector)
            # To allow jit'ing the select function, we need to wrap it in a partial
            slot_select = jax.tree_util.Partial(self.prefill_slot.select)
            # Ask for prefill and insert
            prefill_results, _result_tokens = self.engine.prefill_ex(
                params=self.params,
                padded_tokens=input_ids,
                true_length=true_lengths,
                sampler=slot_select,
            )
            next_token = prefill_results.token.item()
            self.decode_state = self.engine.insert(prefill_results, self.decode_state, slot.id)

            self._post_generate(slot, next_token, generations)
            if not slot.empty:
                prefilled_active_slots.append(slot)

        cached_batch = self._cached_batch(self.batch_id, prefilled_active_slots)
        self.batch_id += 1
        logger.debug("Model ready for decoding")
        return generations, cached_batch

    def _select_from_slots(self, logits: jnp.ndarray, batch_size: int=0) -> jnp.ndarray:
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = logits.shape[0]
        tokens = jnp.full((batch_size, 1), pad_token_id)
        for slot in filter(lambda slot: slot.state == slot.State.READY, self.slots):
            # Every slot might have a different selection criteria, so we are obliged to call select in a loop
            next_token = slot.select(logits[slot.id : slot.id + 1, :])
            tokens = tokens.at[slot.id].set(next_token)
        return tokens

    def decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode the specified prefilled requests.

        Args:
            batches (`List[CachedBatch]`):
                A list of previous batches containing the prefilled requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """

        # In python we should use type duck, but if elements passed on the list are not of the right type, this will
        # prevent raising an error and wasting time. Return an empty generation instead.
        if any(not isinstance(item, CachedBatch) for item in batches):
            logger.error("Unexpected type in decode, expected CachedBatch")
            return [], None

        # batches contains a list composed of ongoing requests:
        # - the batch id returned by the last decode,
        # - the batch id(s) returned by the last prefill(s)
        # Batches are always concatenated during prefill, so we can
        # just carry on with decoding. We adopt the id of the first
        # batch in the list as our next batch id.
        next_batch_id = batches[0].id
        if len(batches) > 1:
            logger.warning("Unexpected multiple batches received, only the first one will be processed.")
        request_ids = []
        for batch in batches:
            request_ids += batch.request_ids
        cleared_request_ids = []
        for slot in self.slots:
            if slot.state == slot.State.READY and slot.request_id not in request_ids:
                cleared_request_ids.append(slot.request_id)
                self.slots.remove(slot)
        if len(cleared_request_ids) > 0:
            logger.info(f"Clearing slot for requests {cleared_request_ids} as they are not requested.")
        active_slots = [slot for slot in self.slots if slot.state == slot.State.READY]
        if len(active_slots) < len(request_ids):
            raise ValueError("Unable to decode tokens for non-prefilled batches (probably due to a previous failure)")

        # Use a custom function to select the next token for each slot
        self.decode_state, result_tokens = self.engine.generate_impl(self.params, self.decode_state, self._select_from_slots)

        generations = []
        for slot in active_slots:
            # Get the next token.
            # Note that for now we ignore is_valid and length as we don't use them, we will re-parse these in post
            # generation.
            next_token = self.decode_state.tokens[slot.id].item()

            if slot.state != Slot.State.READY:
                logger.error(f"Unexpected Slot {slot.id} is not ready for decoding, skipping.")
                raise ValueError("Unexpected Slot is not ready for decoding")

            self._post_generate(slot, next_token, generations)

        cached_batch = self._cached_batch(next_batch_id, active_slots)
        return generations, cached_batch

    def _post_generate(self, slot: Slot, next_token: int, generations: List[Generation]) -> None:
        """Post-generate a slot after the generation has been completed.

        This will check if the slot is finished and append the generated text to the response.

        Args:
            slot (`Slot`):
                The slot to post-generate.
            next_token (`int`):
                The next token generated by the model.
            generations (`List[Generation]`):
                The list of generations to append the slot to.
        """
        # prepare the generation response
        next_token_text = slot.append(next_token)
        generated_text = None
        finish_reason = None
        if next_token == self.tokenizer.eos_token_id:
            finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
        elif slot.stopped:
            # For now we only support the length stopping criteria
            finish_reason = FinishReason.FINISH_REASON_LENGTH
        request_id = slot.request_id
        if finish_reason is not None:
            # We must include the generated text for each finished sequence in the response
            generated_text = GeneratedText(
                text=slot.generated_text, generated_tokens=slot.generated_tokens, finish_reason=finish_reason
            )
            logger.debug(f"Finished generating tokens for request {request_id}")
            # This slot is now empty, it will be removed from the list of
            # active slots.
            slot.clear()
        generations.append(
            Generation(
                request_id=request_id,
                prefill_tokens=None,
                tokens=Tokens(
                    ids=[next_token],
                    logprobs=[0],
                    texts=[next_token_text],
                    is_special=[next_token in self.special_tokens],
                ),
                generated_text=generated_text,
            )
        )

    def _cached_batch(self, batch_id: int, active_slots: List):
        """Create a CachedBatch from the active slots.
        """
        request_ids = [slot.request_id for slot in active_slots if slot.state == Slot.State.READY]
        if len(request_ids) == 0:
            logger.debug("No more pending requests")
            return None
        size = len(request_ids)
        max_tokens = size * self.model.config.sequence_length
        return CachedBatch(id=batch_id, request_ids=request_ids, size=size, max_tokens=max_tokens)

    def filter(self, batch_id: int, keep_request_ids: List[int]) -> CachedBatch:
        """Remove requests that are not listed from the specified batch

        Args:
            batch_id (`int`):
                The id of a cached batch.
            request_ids(`List[int]`):
                The list of requests that must be kept.

        Return:
            A `CachedBatch` containing the pending requests.
        """
        keep_slot_ids = [slot.id for slot in self.slots if slot.request_id in keep_request_ids]
        self._clear(keep_slot_ids)
        return self._cached_batch(batch_id, keep_request_ids)

    def clear(self, batch_id: Optional[int] = None):
        """Remove a subset or all requests from the generator"""
        keep_ids = []
        if batch_id is not None:
            keep_ids = [slot.id for slot in self.slots if slot.batch_id != batch_id]
        return self._clear(keep_ids)

    def _clear(self, keep_slot_ids: List):
        for slot in self.slots:
            if slot.state != Slot.State.EMPTY and slot.id not in keep_slot_ids:
                logger.debug(f"Removing slot {slot.id} with request {slot.request_id}")
                slot.clear()

    @classmethod
    def from_pretrained(
        cls, model_path: str, revision: str, max_batch_size: int, max_sequence_length: int, max_input_tokens: int
    ) -> "TpuGeneratorJetStream":
        """Instantiate a Generator that uses JetStream/Pytorch engine.

        Args:
            model_path (`str`):
                The path to a local model. This path must also contain a Tokenizer.
            revision (`str`):
                Deprecated parameter, only an empty string or None is supported, other values are ignored.
            max_batch_size (`int`):
                The maximum batch size.
            max_sequence_length (`int`):
                The maximum sequence length.
            max_input_tokens (`int`):
                The maximum number of tokens allowed in the input.

        Returns:
            A TpuGenerator.
        """
        if revision != "":
            logger.warning("Revision is not supported for JetStream/Pytorch engine, ignoring.")
        logger.info("Loading model engine (this can take a few minutes).")
        if max_input_tokens > max_sequence_length:
            logger.error("max_input_tokens is greater than max_sequence_length, setting max_sequence_length.")
            raise ValueError("max_input_tokens is greater than max_sequence_length")
        start = time.time()
        torch.set_default_dtype(torch.bfloat16)
        max_output_tokens = max_sequence_length - max_input_tokens
        engine = create_engine(
            model_path,
            max_batch_size,
            sequence_length=max_sequence_length,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        end = time.time()
        logger.info(f"Engine successfully loaded in {end - start:.2f} s.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(engine, tokenizer)
