import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from tempfile import TemporaryDirectory
from typing import List

import docker
import pytest
from aiohttp import ClientConnectorError, ClientOSError, ServerDisconnectedError
from docker.errors import NotFound
from loguru import logger
from test_model import MODEL_CONFIGS
from text_generation import AsyncClient
from text_generation.types import Response


DOCKER_IMAGE = os.getenv("DOCKER_IMAGE", "huggingface/optimum-tpu:latest")
HF_TOKEN = os.getenv("HF_TOKEN", None)
DOCKER_VOLUME = os.getenv("DOCKER_VOLUME", "/data")
TPU_ENV = os.getenv("TPU_ENV")

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def validate_ci_tpu_env_format(env_string: str) -> bool:
    """
    Validate that CI TPU environment string follows '--env Argument' pattern.
    Returns True if valid, False otherwise.
    """
    parts = env_string.split()
    return len(parts) % 2 == 0 and all(
        parts[i] == "--env" and not parts[i + 1].startswith("--env")
        for i in range(0, len(parts), 2)
    )

def process_ci_tpu_env_vars(env_string: str) -> dict:
    """
    Process CI TPU environment string and return dictionary of environment variables.
    """
    env_vars = {}
    # Extract variables from string
    tpu_vars = [x.strip() for x in env_string.split('--env') if x.strip()]

    # Process each variable
    for var in tpu_vars:
        env_value = os.environ.get(var, "")
        env_vars[var] = env_value
        # Log if environment variable is not set
        if not env_value:
            logger.warning(f"TPU environment variable {var} is not set")

    return env_vars


def cleanup_handler(signum, frame):
    logger.info("\nCleaning up containers due to shutdown, please wait...")
    try:
        client = docker.from_env()
        containers = client.containers.list(filters={"name": "tgi-tests-"})
        for container in containers:
            try:
                container.stop()
                container.remove()
                logger.info(f"Successfully cleaned up container {container.name}")
            except Exception as e:
                logger.error(f"Error cleaning up container {container.name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
    sys.exit(1)

signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)

def stream_container_logs(container):
    """Stream container logs in a separate thread."""
    try:
        for log in container.logs(stream=True, follow=True):
            print("[TGI Server Logs] " + log.decode("utf-8"), end="", file=sys.stderr, flush=True)
    except Exception as e:
        logger.error(f"Error streaming container logs: {str(e)}")


class LauncherHandle:
    def __init__(self, port: int):
        self.client = AsyncClient(f"http://localhost:{port}", timeout=3600)

    def _inner_health(self):
        raise NotImplementedError

    async def health(self, timeout: int = 60):
        assert timeout > 0
        start_time = time.time()
        logger.info(f"Starting health check with timeout of {timeout}s")

        for attempt in range(timeout):
            if not self._inner_health():
                logger.error("Launcher crashed during health check")
                raise RuntimeError("Launcher crashed")

            try:
                await self.client.generate("test")
                elapsed = time.time() - start_time
                logger.info(f"Health check passed after {elapsed:.1f}s")
                return
            except (ClientConnectorError, ClientOSError, ServerDisconnectedError) as e:
                if attempt == timeout - 1:
                    logger.error(f"Health check failed after {timeout}s: {str(e)}")
                    raise RuntimeError(f"Health check failed: {str(e)}")
                if attempt % 10 == 0:  # Only log every 10th attempt
                    logger.debug(f"Connection attempt {attempt+1}/{timeout} failed: {str(e)}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error during health check: {str(e)}")
                # Get full traceback for debugging
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise


class ContainerLauncherHandle(LauncherHandle):
    def __init__(self, docker_client, container_name, port: int):
        super(ContainerLauncherHandle, self).__init__(port)
        self.docker_client = docker_client
        self.container_name = container_name

    def _inner_health(self) -> bool:
        try:
            container = self.docker_client.containers.get(self.container_name)
            status = container.status
            if status not in ["running", "created"]:
                logger.warning(f"Container status is {status}")
                # Get container logs for debugging
                logs = container.logs().decode("utf-8")
                logger.debug(f"Container logs:\n{logs}")
            return status in ["running", "created"]
        except Exception as e:
            logger.error(f"Error checking container health: {str(e)}")
            return False


class ProcessLauncherHandle(LauncherHandle):
    def __init__(self, process, port: int):
        super(ProcessLauncherHandle, self).__init__(port)
        self.process = process

    def _inner_health(self) -> bool:
        return self.process.poll() is None


@pytest.fixture(scope="module")
def data_volume():
    tmpdir = TemporaryDirectory()
    yield tmpdir.name
    try:
        # Cleanup the temporary directory using sudo as it contains root files created by the container
        subprocess.run(shlex.split(f"sudo rm -rf {tmpdir.name}"), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error cleaning up temporary directory: {str(e)}")


@pytest.fixture(scope="module")
def launcher(data_volume):
    @contextlib.contextmanager
    def docker_launcher(
        model_id: str,
        trust_remote_code: bool = False,
    ):
        logger.info(f"Starting docker launcher for model {model_id}")
        port = 8080

        client = docker.from_env()

        container_name = f"tgi-tests-{model_id.split('/')[-1]}"

        try:
            container = client.containers.get(container_name)
            logger.info(f"Stopping existing container {container_name}")
            container.stop()
            container.wait()
        except NotFound:
            pass
        except Exception as e:
            logger.error(f"Error handling existing container: {str(e)}")

        model_name = next(name for name, cfg in MODEL_CONFIGS.items() if cfg["model_id"] == model_id)

        args = MODEL_CONFIGS[model_name]["args"].copy()
        if trust_remote_code:
            args.append("--trust-remote-code")

        env = {
            "LOG_LEVEL": "info,text_generation_router,text_generation_launcher=debug",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PJRT_DEVICE": "TPU"
        }
        env.update(MODEL_CONFIGS[model_name]["env_config"].copy())

        # Add model_id to env
        env["MODEL_ID"] = model_id

        if HF_TOKEN is not None:
            env["HF_TOKEN"] = HF_TOKEN

        # Add TPU environment variables when running in CI
        if TPU_ENV:
            logger.info("TPU_ENV is set, adding specific TPU environment variables for the CI")
            logger.debug(f"TPU_ENV: {TPU_ENV}")
            # Validate TPU environment format
            if not validate_ci_tpu_env_format(TPU_ENV):
                raise ValueError("Invalid TPU environment format", TPU_ENV)

            # Process TPU environment variables
            tpu_env_vars = process_ci_tpu_env_vars(TPU_ENV)
            env.update(tpu_env_vars)


        for var in ["MAX_BATCH_SIZE", "HF_SEQUENCE_LENGTH"]:
            if var in os.environ:
                env[var] = os.environ[var]

        volumes = [f"{data_volume}:/data"]

        try:
            # Add debug logging before container creation
            logger.debug(f"Creating container with image {DOCKER_IMAGE}")
            logger.debug(f"Container environment: {env}")
            logger.debug(f"Container volumes: {volumes}")

            # Log equivalent docker run command
            env_str = ' '.join([f'-e {k}="{v}"' for k,v in env.items()])
            volume_str = ' '.join([f'-v {v}' for v in volumes])
            cmd_str = f'docker run -d --name {container_name} {env_str} {volume_str} --shm-size 16G --privileged --ipc host {DOCKER_IMAGE} {" ".join(args)}'
            logger.debug(f"Equivalent docker run command:\n{cmd_str}")

            container = client.containers.run(
                DOCKER_IMAGE,
                command=args,
                name=container_name,
                environment=env,
                auto_remove=False,
                detach=True,
                volumes=volumes,
                shm_size="16G",
                privileged=True,
                ipc_mode="host",
                ports={"80/tcp": 8080}
            )
            logger.info(f"Container {container_name} started successfully")

            # Start log streaming in a background thread
            log_thread = threading.Thread(
                target=stream_container_logs,
                args=(container,),
                daemon=True  # This ensures the thread will be killed when the main program exits
            )
            log_thread.start()

            # Add a small delay to allow container to initialize
            time.sleep(2)

            # Check container status after creation
            status = container.status
            logger.debug(f"Initial container status: {status}")
            if status not in ["running", "created"]:
                logs = container.logs().decode("utf-8")
                logger.error(f"Container failed to start properly. Logs:\n{logs}")

            yield ContainerLauncherHandle(client, container.name, port)

        except Exception as e:
            logger.error(f"Error starting container: {str(e)}")
            # Get full traceback for debugging
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            try:
                container = client.containers.get(container_name)
                logger.info(f"Stopping container {container_name}")
                container.stop()
                container.wait()

                container_output = container.logs().decode("utf-8")
                print(container_output, file=sys.stderr)

                container.remove()
                logger.info(f"Container {container_name} removed successfully")
            except NotFound:
                pass
            except Exception as e:
                logger.error(f"Error cleaning up container: {str(e)}")

    return docker_launcher


@pytest.fixture(scope="module")
def generate_load():
    async def generate_load_inner(client: AsyncClient, prompt: str, max_new_tokens: int, n: int) -> List[Response]:
        try:
            futures = [
                client.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    decoder_input_details=True,
                ) for _ in range(n)
            ]
            return await asyncio.gather(*futures)
        except Exception as e:
            logger.error(f"Error generating load: {str(e)}")
            raise

    return generate_load_inner
