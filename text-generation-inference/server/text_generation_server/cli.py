import os
import sys
from typing import Optional

import typer
from loguru import logger


app = typer.Typer()


@app.command()
def serve(
    model_id: str,
    revision: Optional[str] = None,
    sharded: bool = False,
    trust_remote_code: bool = None,
    uds_path: str = "/tmp/text-generation-server",
    logger_level: str = "INFO",
    json_output: bool = False,
    otlp_service_name: str = "text-generation-inference.server",
    max_input_tokens: Optional[int] = None,
):
    """This is the main entry-point for the server CLI.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.
        sharded (`bool`):
            Whether the model must be sharded or not. Kept for compatibility with the
            text-generation-launcher, but must be set to False.
        trust-remote-code (`bool`):
            Kept for compatibility with text-generation-launcher. Ignored.
        uds_path (`Union[Path, str]`):
            The local path on which the server will expose its google RPC services.
        logger_level (`str`):
            The server logger level. Defaults to *INFO*.
        json_output (`bool`):
            Use JSON format for log serialization.
    """
    if sharded:
        raise ValueError("Sharding is not supported.")
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    if trust_remote_code is not None:
        logger.warning("'trust_remote_code' argument is not supported and will be ignored.")
    if otlp_service_name is not None:
        logger.warning("'otlp_service_name' argument is not supported and will be ignored.")
    if max_input_tokens is not None:
        logger.warning("'max_input_tokens' argument is not supported and will be ignored.")

    # Import here after the logger is added to log potential import exceptions
    from optimum.tpu.model import fetch_model

    from .server import serve

    # Read environment variables forwarded by the launcher
    max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", "1"))
    max_total_tokens = int(os.environ.get("MAX_TOTAL_TOKENS", "64"))

    # Start the server
    model_path = fetch_model(model_id, revision)
    serve(
        model_path,
        revision=revision,
        max_batch_size=max_batch_size,
        max_sequence_length=max_total_tokens,
        uds_path=uds_path
    )


@app.command()
def download_weights(
    model_id: str,
    revision: Optional[str] = None,
    logger_level: str = "INFO",
    json_output: bool = False,
    auto_convert: Optional[bool] = None,
    extension: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
):
    """Download the model weights.

    This command will be called by text-generation-launcher before serving the model.
    """
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level=logger_level,
        serialize=json_output,
        backtrace=True,
        diagnose=False,
    )

    if extension is not None:
        logger.warning("'extension' argument is not supported and will be ignored.")
    if trust_remote_code is not None:
        logger.warning("'trust_remote_code' argument is not supported and will be ignored.")
    if auto_convert is not None:
        logger.warning("'auto_convert' argument is not supported and will be ignored.")

    # Import here after the logger is added to log potential import exceptions
    from optimum.tpu.model import fetch_model

    fetch_model(model_id, revision)
