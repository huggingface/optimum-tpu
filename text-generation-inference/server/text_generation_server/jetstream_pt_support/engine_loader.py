# Import torch_xla2 first
import torch_xla2  # isort:skip

from typing import TYPE_CHECKING, Any

import jax
from jetstream_pt import fetch_models, torchjax
from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.environment import (
    JetEngineEnvironment,
    JetEngineEnvironmentData,
    QuantizationConfig,
)
from loguru import logger


if TYPE_CHECKING:
    from transformers import PretrainedConfig
from transformers import AutoConfig

from .compatibility import model_can_use_jetstream_pt
from .models import GemmaModel, LlamaModel, MixtralModel


def _get_head_dim(config: "PretrainedConfig") -> int:
    if hasattr(config, "head_dim"):
        return config.head_dim
    return config.hidden_size // config.num_attention_heads

def load_model_info(config: "PretrainedConfig") -> Any:
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = _get_head_dim(config)
    num_kv_heads = config.num_key_value_heads
    n_reps = num_heads // num_kv_heads
    if config.model_type == "llama":
        model_class = LlamaModel
    elif config.model_type == "gemma":
        model_class = GemmaModel
    elif config.model_type == "mixtral":
        model_class = MixtralModel
    else:
        raise ValueError(f"Unsupported model type {config.model_type}")
    model_info = fetch_models.ModelInfo(
        model_class=model_class,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        n_reps=n_reps,
    )
    return model_info


def create_engine_env_data(
    model_path: str,
    batch_size: int,
    sequence_length: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> Any:
    if not model_can_use_jetstream_pt(model_path):
        return None
    # First get config
    config = AutoConfig.from_pretrained(model_path)
    model_info = load_model_info(config)
    if model_info is None:
        return None

    shard_on_batch = False
    max_cache_length = max_input_tokens + max_output_tokens

    logger.info(f"Creating engine with max_cache_length={max_cache_length} = {max_input_tokens} + {max_output_tokens}")
    env_data = JetEngineEnvironmentData(
        tokenizer_path="", # Tokenizer is not user, HF tokenizer is used instead
        checkpoint_path=model_path,
        checkpoint_format="safetensors",
        batch_size=batch_size,
        max_decode_length=sequence_length,
        max_input_sequence_length=max_input_tokens,
        quant_config=QuantizationConfig(),
        cache_sequence_length=max_cache_length,
        bf16_enable=True,
        sharding_config_path="",
        shard_on_batch=shard_on_batch,
        n_reps=model_info.n_reps,
    )
    env_data.cache_shape = (
        batch_size,
        model_info.num_kv_heads,
        max_cache_length,
        model_info.head_dim,
    )
    env_data.num_layers = model_info.num_layers
    return env_data


def instantiate_model_from_repo_id(
    model_dir: str,
    env: Any,
):
    """Create model instance by hf model_dir, and its config"""
    config = AutoConfig.from_pretrained(model_dir)
    model_info = load_model_info(config)
    # at this point we can be quite optimistic and just assert
    assert model_info is not None

    env.device = "meta"
    model = model_info.model_class.from_config(config, env)
    weights = fetch_models._load_weights(model_dir)
    weights = model.convert_hf_weights(weights)

    model.load_state_dict(weights, assign=True, strict=False)

    return model


def shard_weights(env, weights, weight_shardings):
    """Shard weights according to weight_shardings"""
    for k, v in weight_shardings.items():
        logger.debug(f"SHARDING {k} {v}")
    sharded = {}
    for key, val in weights.items():
        sharding = env.sharding_by_axis(weight_shardings.get(key, -1))
        with jax.default_device(jax.devices("cpu")[0]):
            # Note we clone to avoid a core-dump that might happen otherwise when calling device_put
            arr = torch_xla2.tensor.t2j(val.clone())
        arr = jax.device_put(arr, sharding)
        sharded[key] = torchjax.to_torch(arr)
    return sharded


def create_engine(
    model_path: str,
    batch_size: int,
    sequence_length: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> PyTorchEngine:
    # NOTE: for now no quantization is done
    env_data = create_engine_env_data(model_path, batch_size, sequence_length, max_input_tokens, max_output_tokens)
    if env_data is None:
        return None

    env = JetEngineEnvironment(env_data)
    model = instantiate_model_from_repo_id(model_path, env)
    # Update config with engine data
    model.config.batch_size = batch_size
    model.config.sequence_length = sequence_length

    weight_shardings = model.get_sharding_annotations()
    sharded_weights = shard_weights(env, model.state_dict(), weight_shardings)

    return PyTorchEngine(
        pt_model=model,
        env=env,
        weights=torchjax.from_torch_with_copy(sharded_weights),
    )
