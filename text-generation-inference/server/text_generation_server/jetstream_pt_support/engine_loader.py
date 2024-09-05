# Import torch_xla2 first
import torch_xla2  # isort:skip

from typing import Any

import jax
from jetstream_pt import fetch_models, torchjax
from jetstream_pt.environment import (
    JetEngineEnvironment,
    JetEngineEnvironmentData,
    QuantizationConfig,
)
from loguru import logger
from transformers import AutoConfig, PretrainedConfig

from .engine import HfEngine
from .llama_model_exportable_hf import TransformerHf


def load_llama_model_info(config: PretrainedConfig) -> Any:
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads
    n_reps = num_heads // config.num_key_value_heads
    model_info = fetch_models.ModelInfo(
        TransformerHf,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        n_reps=n_reps,
    )
    return model_info


def load_model_info(config: PretrainedConfig) -> Any:
    # For now only Llama is supported
    if config.model_type == "llama":
        return load_llama_model_info(config)
    # Other models supports can be added here later
    return None


def create_engine_env_data(
    model_path: str,
    batch_size: int,
    sequence_length: int,
    max_input_tokens: int,
    max_output_tokens: int,
) -> Any:
    # First get config
    config = AutoConfig.from_pretrained(model_path)
    model_info = load_model_info(config)
    if model_info is None:
        return None

    shard_on_batch = False
    max_cache_length = max_input_tokens + max_output_tokens

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
        config.num_key_value_heads,
        max_cache_length,
        model_info.head_dim,
    )
    env_data.num_layers = model_info.num_layers
    return env_data


def create_model(model_path: str, env: Any) -> Any:
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == "llama":
        return TransformerHf.from_config(config, env)


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
    model = create_model(model_dir, env)
    weights = fetch_models._load_weights(model_dir)
    updated_keys = model.get_hf_names_to_real_name()
    for name, updated in updated_keys.items():
        if name in weights:
            val = weights.pop(name)
            weights[updated] = val

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
) -> HfEngine:
    # NOTE: for now no quantization is done
    env_data = create_engine_env_data(model_path, batch_size, sequence_length, max_input_tokens, max_output_tokens)
    if env_data is None:
        return None

    env = JetEngineEnvironment(env_data)
    model = instantiate_model_from_repo_id(model_path, env)
    weight_shardings = model.get_sharding_annotations()
    sharded_weights = shard_weights(env, model.state_dict(), weight_shardings)

    return HfEngine(
        pt_model=model,
        env=env,
        weights=torchjax.from_torch_with_copy(sharded_weights),
    )
