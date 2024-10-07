import copy
import logging
import os
import sys
import time
import traceback
from bisect import bisect_left
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.generation import GenerationConfig

import optimum.tpu.xla_logger as logger
from optimum.tpu import AutoModelForCausalLM
from optimum.tpu.generation import TokenSelector
from optimum.tpu.static_cache_xla import StaticCacheXla
from optimum.tpu.xla_mp_comm import AgentMailbox, RootMailbox

from .generator_base import Generator
from .pb.generate_pb2 import (
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


# Disable optimum-tpu warnings as it seems to block the server after a while
optimum_logger = logging.getLogger("optimum.tpu")
optimum_logger.setLevel("CRITICAL")

# These will do some bucketing on prefill lengths to avoid too many different sizes
PREFILL_LENGTHS = list(range(6, 16)) + [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]

def take_nearest_length(length: int) -> int:
  """Gets the nearest length to the right in a set of lengths."""
  pos = bisect_left(PREFILL_LENGTHS, length)
  if pos == len(PREFILL_LENGTHS):
    return PREFILL_LENGTHS[-1]
  return PREFILL_LENGTHS[pos]

class Slot:
    """Represents a slot in a static batch"""

    class State(Enum):
        EMPTY = 0
        PAUSE = 1
        READY = 2

    def __init__(self, id: int, tokenizer: PreTrainedTokenizerBase, device: [str, torch.device]):
        self._id = id
        self._tokenizer = tokenizer
        self.clear()
        self._device = device

    def clear(self):
        """Clear the slot and mark it as available."""
        self._state = Slot.State.EMPTY
        self._batch_id = None
        self._request_id = None
        self._inputs = ""
        self._generation_config = None
        self._tokens = []
        self._mask = None
        self._selector = None
        self._generated_tokens = 0
        self._next_text_token_start = 0
        self._next_text_token_end = 0
        self._generated_text = ""
        self._next_text = ""
        self._kv_cache = None
        self._truncate = 0
        self._position_id = 0

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
    def cached_text(self) -> str:
        return self._inputs + self._generated_text

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
    def position_id(self) -> int:
        return self._position_id

    @position_id.setter
    def position_id(self, cur_pos: int):
        self._position_id = cur_pos

    @property
    def cache_position(self) -> int:
        # This corresponds to the cache position for this slot
        return self._next_text_token_start


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
        self.seed = request.parameters.seed
        # TODO: watermark
        self._generation_config.max_new_tokens = request.stopping_parameters.max_new_tokens
        self._max_new_tokens = self._generation_config.max_new_tokens
        # TODO: stop_sequences, ignore_eos_token

    def reset(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, selector: TokenSelector):
        """Reset the slot for the next generation.

        Args:
            input_ids: (`torch.LongTensor`):
                The new input_ids to use to generate the next token.
            attention_mask: (`torch.LongTensor`):
                The new attention_mask to use to generate the next token.
            selector: (`TokenSelector`):
                An object implementing the updated token selection logic.
        """
        self._tokens = input_ids.cpu()
        self._next_text_token_start = 0
        self._next_text_token_end = torch.numel(self._tokens)
        self._next_text = ""
        if attention_mask is not None:
            self._mask = attention_mask.clone()
        else:
            self._mask = None
        self._selector = selector

    def pause(self):
        """Mark the current slot as paused for generation.

        Note that the KV cache for this slot will still be filled.
        """
        # Drop the last token as it will be added back when resuming the slot
        self._generated_tokens -= 1
        # Since generated tokens are now part of the prefill, we need to reevaluate
        # max_new_tokens for the next generation
        self._generation_config.max_new_tokens = self._max_new_tokens - self._generated_tokens
        self._state = Slot.State.PAUSE

    def resume(self):
        """Mark the slot as ready for generation."""
        self._state = Slot.State.READY

    def _decode_next_tokens(
        self,
    ) -> str:
        """Hack to hopefully support generate_stream for the maximum number of tokenizers"""
        # Copy the tokens to CPU to avoid recompilation on TPU. Post-processing is quite fast anyway.
        tokens = self._tokens.cpu()
        # We need to include the tokens that produced the last text to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        new_text = self._tokenizer.decode(tokens[self._next_text_token_start :], skip_special_tokens=False)
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
        self._next_text_token_end = torch.numel(tokens)
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
        self._tokens = torch.cat([self._tokens, torch.tensor([next_token], dtype=self._tokens.dtype)])
        # Update mask only if it was set previously
        if self._mask is not None:
            self._mask = torch.cat([self._mask, torch.tensor([1], dtype=self._mask.dtype)])
        self._generated_tokens += 1
        next_text = self._decode_next_tokens()
        # Now that a new token has been generated, we can append the previous one to the generated text
        self._generated_text += self._next_text
        self._next_text = next_text
        return next_text

    def select(self, input_ids: torch.LongTensor, logits: torch.Tensor) -> torch.LongTensor:
        """Select the next token from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A scalar torch.LongTensor` containing the selected token.
        """
        return self._selector.select(input_ids, logits)[0]

    @property
    def stopped(self) -> bool:
        # unsqueeze tokens to avoid problems with stopping criteria
        tokens = self._tokens.unsqueeze(0)
        return bool(torch.all(self._selector.stopping_criteria(tokens, None)))

    @property
    def generated_text(self) -> str:
        return self._generated_text + self._next_text

    @property
    def next_token(self) -> int:
        return None if len(self._tokens) == 0 else self._tokens[-1]

    @property
    def attention_mask(self) -> torch.LongTensor:
        return self._mask

    @property
    def max_token(self) -> int:
        return self._generation_config.max_length


class TpuGeneratorSingleThread(Generator):
    """A Generator for models running on TPU, single threaded."""

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.model = model
        # Specify padding options for decoder-only architecture
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        self.tokenizer = tokenizer
        self.special_tokens = self.tokenizer.all_special_ids
        # Slots are empty to begin with, they will be populated as new batches arrive
        self.slots = []
        self.batch_id = 0
        # Note: this index will _never_ be decremented, and that's fine.
        self.slot_index = 0
        self.past_key_values = None
        # _supports_static_cache is specific to some models (e.g.: Gemma and Llama).
        self._supports_static_cache = True
        if getattr(self.model, "_supports_static_cache", False) is False:
            logger.warning(
                f"Static cache not available for {self.model.__class__.__name__}. Performance will be affected"
            )
            self._supports_static_cache = False
        # compile model when possible to accelerate decoding
        if model.device.type == "xla" and ("DBG_COMPILE" in os.environ):
            self.model_one_token = torch.compile(model, backend="openxla")
            logger.debug("Model compiled for decoding")
        else:
            self.model_one_token = model

    @property
    def info(self) -> InfoResponse:
        """Returns the expected InfoResponse."""
        dtype = getattr(self.model.config, "torch_dtype", "float32")
        return InfoResponse(
            requires_padding=True,
            dtype=str(dtype),
            device_type="xla",
        )

    def _create_dummy_request(self, max_tokens: int) -> Batch:
        """Create a dummy request for warmup."""
        # Generate a random input with slightly more tokens than requested, because special tokens are going to be
        # skipped.
        MARGIN = 10
        input_tokens = torch.randint(self.model.config.vocab_size, (1, max_tokens + MARGIN), dtype=torch.int64)
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
        # NOTE: later self.model.config.batch_size might become self.model.config.max_batch_size.
        if self.model.config.batch_size is not None:
            batch_size = self.model.config.batch_size
        else:
            # batch size is not set, just assume it's unlimited and accept all requests
            batch_size = len(batch.requests)
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
        bucket_seq_len = take_nearest_length(seq_len)
        requests = [self._create_dummy_request(seq_len) for _ in range(batch_size)]
        for _ in reversed(range(batch_size)):
            # Prefill with different truncate sizes to test all prefill lengths. List is reversed so first longest
            # sequences are tested and, if there is a memory failure, that will appear sooner.
            for l in reversed(PREFILL_LENGTHS):
                # Skip all the unsupported lengths
                if l > bucket_seq_len:
                    continue
                # Set all truncate values for all requests
                for r in requests:
                    r.truncate = l
                    r.stopping_parameters.max_new_tokens = 10
                warmup_batch = Batch(id=0,
                                     requests=requests,
                                     size=len(requests),
                                     max_tokens=batch.max_tokens)
                logger.debug(f"Warmup for {len(requests)} requests, truncate value {l} seq_len {seq_len}")
                _generations, next_batch = self.prefill(warmup_batch)
                if next_batch is not None:
                    self.decode([next_batch])
                else:
                    logger.debug(f"No decode on warmup for {len(requests)}x{l}")
                self.clear()
            # remove the last requests to decrease the batch size
            requests.pop()

        elapsed = time.time() - start
        logger.debug(f"Warmup done, took {elapsed:.2f}s")
        return batch_size * seq_len

    @torch.no_grad
    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        """Prefill new requests.

        Args:
            batch (`Batch`):
                A batch containing the new requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        slots = {state: [] for state in Slot.State}
        for slot in self.slots:
            slots[slot.state].append(slot)
        active_slots = slots[Slot.State.READY]
        # Delete all empty slots, no need to have them anymore
        empty_slots = slots[Slot.State.EMPTY]
        model_batch_size = self.model.config.batch_size
        if model_batch_size is not None and model_batch_size < len(active_slots) + len(batch.requests):
            # If raising an error here wouldn't crash the server, we could raise a ValueError
            error = ValueError(
                f"Cannot prefill {len(batch.requests)} new request(s)."
                f" Maximum batch size supported is: {model_batch_size}."
            )
            # but since it's not possible, we just log the error and return an empty generation
            logger.error(error)
            return [], None
        for slot in empty_slots:
            self.slots.remove(slot)
        # Assign each request to an empty slot
        logger.debug(f"Prefilling {len(batch.requests)} new request(s) adding to {len(active_slots)} active slot(s)")
        for request in batch.requests:
            # Dynamically create a new slot for each request
            slot = Slot(self.slot_index, self.tokenizer, self.model.device)
            self.slot_index += 1
            slot.assign(self.batch_id, request, self.model.generation_config)
            self.slots.append(slot)
            logger.debug(f"Request {slot.request_id} assigned to slot {slot.id}")
        # Reconstruct the full inputs (without padding) as seen by the model.
        # This comprises:
        # - the inputs for new requests,
        # - the inputs and the generated text that has already been cached (i.e. excluding the last generated token)
        #   for unfinished requests.

        # Prepare inputs. They need to be tokenized and truncated afterwards.
        max_len = 0
        batch_inputs = []
        for slot in self.slots:
            batch_inputs.append(slot.cached_text)
            max_len = max(max_len, slot.truncate)
        if max_len == 0:
            max_len = self.model.config.sequence_length
        tokenized_inputs = self.tokenizer(batch_inputs,
                                          return_tensors="pt",
                                          padding=True,
                                          truncation=True,
                                          max_length=max_len)
        seq_length = tokenized_inputs.input_ids.size(-1)
        seq_length = min(seq_length, self.model.config.sequence_length)
        batch_size = len(self.slots)
        # Initialize input_ids and attention_mask with padding (to make them all the same size)
        input_ids = torch.full((batch_size, seq_length), self.tokenizer.pad_token_id, dtype=torch.int64)
        attention_mask = torch.full((batch_size, seq_length), 0, dtype=torch.int64)

        # Pause previously active slots during generation and store their last token.
        next_tokens = []
        for slot in active_slots:
            next_tokens.append(slot.next_token)
            slot.pause()
        # Each slot must be reset with the padded inputs and masks
        for i, slot in enumerate(self.slots):
            assert slot.state != slot.state.EMPTY

            truncation = min(tokenized_inputs.input_ids.size(-1), input_ids.size(-1))
            if slot.truncate > 0:
                truncation = min(truncation, slot.truncate)
            input_ids[i, -truncation:] = tokenized_inputs.input_ids[i, -truncation:]
            slot_input_ids = input_ids[i : i + 1, :]
            # Padded input ids are also required to set logits processors and stopping criterias
            try:
                selector = TokenSelector.create(
                    slot_input_ids,
                    slot.generation_config,
                    self.model,
                    self.model.config.sequence_length,
                    seed=slot.seed,
                )
            except ValueError as e:
                # This is very unlikely, but it seems it could be possible if router does not check values beforehand.
                # In that case, we just skip the slot, and mark it as empty. This should prevent returning this to the
                # router.
                logger.error(f"Invalid generation parameters for slot {slot.id}. Skipping it. Error: {e}")
                slot.clear()
                continue
            slot_input_ids = slot_input_ids.squeeze(dim=0).type(torch.int64)
            attention_mask[i, -truncation:] = tokenized_inputs.attention_mask[i, -truncation:]
            if self._supports_static_cache:
                # Attention mask does not need to be tracked when using static cache
                slot_attention_mask = None
            else:
                slot_attention_mask = attention_mask[i]
            slot.reset(slot_input_ids, slot_attention_mask, selector)
        # Clear KV cache
        self.past_key_values = None
        # Obtain position ids using attention mask.
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
        # Save position id for every slot
        for slot, position_id in zip(self.slots, position_ids):
            slot.position_id = position_id.max().item() + 1

        extra_args = {}
        if self._supports_static_cache:
            self.past_key_values = StaticCacheXla(
                config=self.model.config,
                max_batch_size=len(self.slots),
                max_cache_len=self.model.config.sequence_length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            extra_args["cache_position"] = torch.arange(seq_length, device=self.model.device)
            extra_args["past_key_values"] = self.past_key_values
        else:
            # Reset/clear KV cache
            self.past_key_values = None
        generation, next_batch = self._generate_token(
            self.batch_id,
            input_ids.to(self.model.device),
            self.model,
            attention_mask=attention_mask.to(self.model.device),
            position_ids=position_ids.to(self.model.device),
            **extra_args,
        )
        self.batch_id += 1

        # Reactivate previously active slots for the next decode, and append
        # back their next token.
        for slot, next_token in zip(active_slots, next_tokens):
            slot.append(next_token)
            slot.resume()
        logger.debug("Model ready for decoding")
        return generation, next_batch

    @torch.no_grad
    def decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        """Decode the specified prefilled requests.

        Args:
            batches (`List[CachedBatch]`):
                A list of previous batches containing the prefilled requests.

        Return:
            A list of `Generation` for each request and a `CachedBatch` containing all pending requests.
        """
        # batches contains a list composed of:
        # - the batch id returned by the last decode,
        # - the batch id(s) returned by the last prefill(s)
        # Batches are always concatenated during prefill, so we can
        # just carry on with decoding. We adopt the id of the first
        # batch in the list as our next batch id.
        next_batch_id = batches[0].id
        request_ids = []
        for batch in batches:
            request_ids += batch.request_ids
        cleared_request_ids = []
        for slot in self.slots:
            if slot.state == slot.State.READY and slot.request_id not in request_ids:
                cleared_request_ids.append(slot.request_id)
                slot.clear()
        if len(cleared_request_ids) > 0:
            logger.info(f"Clearing slot for requests {cleared_request_ids} as they are not requested.")
        active_slots = [slot for slot in self.slots if slot.state == slot.State.READY]
        if len(active_slots) < len(request_ids):
            logger.error("Unable to decode tokens for non-prefilled batches (probably due to a previous failure)")
        # Reconstruct input_ids and attention_mask from slots
        input_ids = None
        attention_mask = None
        batch_size = len(self.slots)
        position_ids = torch.zeros(
            [batch_size, 1],
            dtype=torch.int64,
        )
        # init pad_token_id and input_ids
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            if isinstance(self.tokenizer.eos_token_id, list):
                pad_token_id = self.tokenizer.eos_token_id[0]
            else:
                pad_token_id = self.tokenizer.eos_token_id
        # Create blank inputs covering all slots (even empty ones)
        input_ids = torch.full(
            [batch_size, 1],
            fill_value=pad_token_id,
            dtype=torch.int64,
        )
        cache_position = torch.zeros([1], dtype=torch.int64)
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.EMPTY:
                # input_ids are simply the tokens generated by the last decode or prefill requests (other tokens are cached)
                input_ids.index_put_([torch.tensor([i])], slot.next_token)
                if not self._supports_static_cache:
                    # When using dynamic cache, the whole attention mask needs to be passed over to the model at each iteration.
                    if attention_mask is None:
                        # Create default mask covering all slots (even empty ones)
                        attention_mask = torch.zeros(
                            [batch_size, slot.attention_mask.size(-1)],
                            dtype=torch.int64,
                        )
                    attention_mask.index_put_([torch.tensor([i])], slot.attention_mask)
                position_ids.index_put_([torch.tensor([i])], torch.tensor(slot.position_id))
                cache_position = torch.maximum(cache_position, torch.tensor([slot.cache_position]))
        if input_ids is None:
            raise ValueError("Unable to decode tokens for non-prefilled batches (probably due to a previous failure)")
        extra_args = {}
        if self._supports_static_cache:
            extra_args["cache_position"] = position_ids.max().unsqueeze(0).to(self.model.device)
        else:
            extra_args["attention_mask"] = attention_mask.to(self.model.device)
        extra_args["past_key_values"] = self.past_key_values
        generations, next_batch = self._generate_token(
            next_batch_id,
            input_ids.to(self.model.device),
            self.model_one_token,
            position_ids=position_ids.to(self.model.device),
            **extra_args,
        )
        for slot, gen in zip(self.slots, generations):
            slot.position_id += len(gen.tokens.ids)

        return generations, next_batch

    def _generate_token(
        self, next_batch_id: int, input_ids: torch.LongTensor, model: torch.nn.Module, **forward_extra_params
    ) -> Tuple[List[Generation], CachedBatch]:
        # Add barrier to allow next graph step to always be the same
        xm.mark_step()
        # Forward
        outputs = model(
            input_ids,
            return_dict=True,
            use_cache=True,
            **forward_extra_params,
        )
        if not self._supports_static_cache:
            # Save KV cache
            self.past_key_values = outputs.past_key_values
        # Barrier for XLA model
        xm.mark_step()
        ret = self._post_generate(outputs, next_batch_id, input_ids)
        return ret

    def _post_generate(
        self, outputs: Dict, next_batch_id: int, input_ids: torch.LongTensor
    ) -> Tuple[List[Generation], CachedBatch]:
        generations = []
        active_slots = False
        for i, slot in enumerate(self.slots):
            if slot.state != Slot.State.READY:
                continue
            request_id = slot.request_id
            next_token_logits = outputs.logits[i : i + 1, -1, :]
            slot_input_ids = input_ids[i : i + 1, :]
            next_token = slot.select(slot_input_ids, next_token_logits)
            next_token_text = slot.append(next_token)
            generated_text = None
            finish_reason = None
            if next_token == self.tokenizer.eos_token_id:
                finish_reason = FinishReason.FINISH_REASON_EOS_TOKEN
            elif slot.stopped:
                # For now we only support the length stopping criteria
                finish_reason = FinishReason.FINISH_REASON_LENGTH
            if finish_reason is not None:
                # We must include the generated text for each finished sequence in the response
                generated_text = GeneratedText(
                    text=slot.generated_text, generated_tokens=slot.generated_tokens, finish_reason=finish_reason
                )
                logger.debug(f"Finished generating tokens for request {request_id}")
                # This slot is now empty, it will be removed from the list of
                # active slots once a new prefill is requested
                slot.clear()
            else:
                active_slots = True
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
        batch = None
        if active_slots:
            # Whatever initial batch these requests came from, we always return all pending requests in a single batch
            request_ids = [slot.request_id for slot in self.slots if slot.state == Slot.State.READY]
            batch = self._cached_batch(next_batch_id, request_ids)
        else:
            logger.debug("No more pending requests")
        return generations, batch

    def _cached_batch(self, batch_id: int, request_ids: List):
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
    def from_pretrained(cls, model_path: str, revision: str, max_batch_size: int, max_sequence_length: int):
        """Instantiate a TpuGenerator.

        Args:
            model_path (`str`):
                The path to a local model. This path must also contain a Tokenizer.
            revision (`str`):
                The revision of the model.
            max_batch_size (`int`):
                The maximum batch size.
            max_sequence_length (`int`):
                The maximum sequence length.

        Returns:
            A TpuGenerator.
        """
        logger.info("Loading model (this can take a few minutes).")
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, batch_size=max_batch_size, sequence_length=max_sequence_length
        )
        end = time.time()
        logger.info(f"Model successfully loaded in {end - start:.2f} s.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(model, tokenizer)


class GeneratorCommand(Enum):
    INFO = 0
    WARMUP = 1
    PREFILL = 2
    DECODE = 3
    FILTER = 4
    CLEAR = 5
    DELETE = -1


def _mp_fn(
    rank, model_path: str, revision: str, max_batch_size: int, max_sequence_length: int, root_mailbox: RootMailbox
):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    # create agent mailbox out of root's one
    mailbox = AgentMailbox(root_mailbox)

    # re-init logger for each child process
    logger_level = os.environ.get("LOGGER_LEVEL_GENERATOR", "DEBUG")
    logger.logger.remove()
    logger.logger.add(
        sys.stdout,
        filter="text_generation_server",
        level=logger_level,
        backtrace=True,
        diagnose=False,
    )

    logger.debug(
        f"Rank {rank} on {device} real device {xm.xla_real_devices([device])} ordinal {xm.get_ordinal()} "
        + f"world size {world_size}"
    )

    generator = TpuGeneratorSingleThread.from_pretrained(model_path, revision, max_batch_size, max_sequence_length)
    # TODO: maybe model_config can be removed from mailbox

    def return_to_caller(*data):
        # consider adding a rendezvous here
        if rank == 0:
            xm.mark_step()
            mailbox.send(*data)

    while True:
        xm.rendezvous("start")
        if rank == 0:
            mailbox.agent_ready.set()
            mailbox.receive()
        # Wait for rank 0 to receive command
        xm.rendezvous("wait_command")
        command, data = mailbox.command_data
        logger.debug(f"Generator@{rank} {command.name}")
        try:
            if command == GeneratorCommand.INFO:
                info = generator.info
                return_to_caller(info.SerializeToString())
            if command == GeneratorCommand.WARMUP:
                batch = Batch.FromString(data[0])
                return_to_caller(generator.warmup(batch=batch))
            if command == GeneratorCommand.PREFILL:
                batch = Batch.FromString(data[0])
                generations, cached_batch = generator.prefill(batch=batch)
                s_cached_batch = cached_batch.SerializeToString() if cached_batch is not None else None
                return_to_caller([g.SerializeToString() for g in generations], s_cached_batch)
            if command == GeneratorCommand.DECODE:
                batches = [CachedBatch.FromString(b) for b in data[0]]
                generations, cached_batch = generator.decode(batches=batches)
                s_cached_batch = cached_batch.SerializeToString() if cached_batch is not None else None
                return_to_caller([g.SerializeToString() for g in generations], s_cached_batch)
            if command == GeneratorCommand.FILTER:
                batch_id, request_ids = data
                cached_batch = generator.filter(batch_id, request_ids)
                return_to_caller(cached_batch.SerializeToString())
            if command == GeneratorCommand.CLEAR:
                batch_id = data[0]
                generator.clear(batch_id)
            if command == GeneratorCommand.DELETE:
                if rank == 0:
                    # Set agent to ready
                    mailbox.agent_ready.set()
                break
        except Exception as e:
            logger.error(f"Error in command {command.name}")
            mailbox.agent_error.set()
            mailbox.agent_ready.set()
            exc_info = sys.exc_info()
            logger.error(''.join(traceback.format_exception(*exc_info)))
            raise e
        # If error was only happening on one of the threads, all of them should exit
        if mailbox.agent_error.is_set():
            return


def model_loop_fn(*args):
    """Spawn processes in the TPUs forwarding arguments"""
    xmp.spawn(_mp_fn, args=(args), join=True, daemon=False)


class TpuGenerator(Generator):
    """A Generator for models running on TPU.

    This generator actually spawns several processes to handle the requests in sharded models whenever possible.
    """

    def __init__(self, model_path: str, revision: str, max_batch_size: int, max_sequence_length: int):
        manager = mp.Manager()
        self.mailbox = RootMailbox(manager)

        # Disable parallelism on tokenizers to avoid deadlocks on TPU threads
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_loop = mp.Process(
            target=model_loop_fn, args=(model_path, revision, max_batch_size, max_sequence_length, self.mailbox)
        )
        self.model_loop.start()

    @property
    def info(self) -> InfoResponse:
        s_info = self.mailbox.send(GeneratorCommand.INFO, None)[0]
        return InfoResponse.FromString(s_info)

    def warmup(self, batch: Batch) -> int:
        return self.mailbox.send(GeneratorCommand.WARMUP, batch.SerializeToString())[0]

    def prefill(self, batch: Batch) -> Tuple[List[Generation], CachedBatch]:
        s_generations, s_cached_batch = self.mailbox.send(GeneratorCommand.PREFILL, batch.SerializeToString())
        generations = [Generation.FromString(g) for g in s_generations]
        cached_batch = CachedBatch.FromString(s_cached_batch) if s_cached_batch is not None else None
        return generations, cached_batch

    def decode(self, batches: List[CachedBatch]) -> Tuple[List[Generation], CachedBatch]:
        s_batches = [b.SerializeToString() for b in batches]
        s_generations, s_cached_batch = self.mailbox.send(GeneratorCommand.DECODE, s_batches)
        generations = [Generation.FromString(g) for g in s_generations]
        cached_batch = CachedBatch.FromString(s_cached_batch) if s_cached_batch is not None else None
        return generations, cached_batch

    def filter(self, batch_id: int, request_ids: List[int]) -> CachedBatch:
        s_cached_batch = self.mailbox.send(GeneratorCommand.FILTER, batch_id, request_ids)[0]
        return CachedBatch.FromString(s_cached_batch)

    def clear(self, batch_id: Optional[int] = None):
        self.mailbox.send(GeneratorCommand.CLEAR, batch_id)

    def leave(self):
        if self.mailbox is None:
            return
        self.mailbox.send(GeneratorCommand.DELETE)
        # Use Loguru's logger directly, to avoid errors whyle TPU is shutting down
        logger.logger.debug("Joining...")
        self.model_loop.join()
        logger.logger.debug("Generator loop finished")
        self.mailbox = None

    @property
    def config(self):
        return self.mailbox.config

    def __del__(self):
        self.leave()

    @classmethod
    def from_pretrained(cls, model_path: str, revision: str, max_batch_size: int, max_sequence_length: int):
        """Instantiate a Generator distributed on as many cores as possible.

        Args:
            model_path (`str`):
                The path to a local model. This path must also contain a Tokenizer.
            revision (`str`):
                The revision of the model.
            max_batch_size (`int`):
                The maximum batch size.
            max_sequence_length (`int`):
                The maximum sequence length.

        Returns:
            A TpuGenerator.
        """
        return cls(model_path, revision, max_batch_size, max_sequence_length)
