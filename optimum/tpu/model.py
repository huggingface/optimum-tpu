import os
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME


def get_export_kwargs_from_env():
    batch_size = os.environ.get("HF_BATCH_SIZE", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    sequence_length = os.environ.get("HF_SEQUENCE_LENGTH", None)
    if sequence_length is not None:
        sequence_length = int(sequence_length)
    return {
        "task": "text-generation",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
    }


def fetch_model(
    model_id: str,
    revision: Optional[str] = None,
) -> str:
    """Fetch a model to local cache.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.

    Returns:
        Model ID or path of the model available in cache.
    """
    if os.path.isdir(model_id):
        if revision is not None:
            logger.warning("Revision {} ignored for local model at {}".format(revision, model_id))
        return model_id

    # Download the model from the Hub (HUGGING_FACE_HUB_TOKEN must be set for a private or gated model)
    # Note that the model may already be present in the cache.
    start = time.time()
    local_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["*.json", "model*.safetensors", SAFE_WEIGHTS_INDEX_NAME, "tokenizer.*"],
    )
    end = time.time()
    logger.info(f"Model successfully fetched in {end - start:.2f} s.")

    # This will allow to set config to update specific config such as
    # batch_size and sequence_length.
    export_kwargs = get_export_kwargs_from_env()
    config = AutoConfig.from_pretrained(local_path)
    config.update(export_kwargs)
    config.save_pretrained(local_path)
    end = time.time()
    logger.info(f"Model config updated in {end - start:.2f} s.")

    return Path(local_path)
