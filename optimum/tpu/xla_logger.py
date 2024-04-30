import torch_xla.core.xla_model as xm
from loguru import logger


"""
This is just a shallow wrapper to loguru's logger, to only log messages on the master ordinal and avoid repeating
messages on all the other ordinals threads.
"""

def warning(message: str):
    if xm.get_ordinal() == 0:
        logger.opt(depth=1).warning(message)

def info(message: str):
    if xm.get_ordinal() == 0:
        logger.opt(depth=1).info(message)

def debug(message: str):
    if xm.get_ordinal() == 0:
        logger.opt(depth=1).debug(message)

def error(message: str):
    if xm.get_ordinal() == 0:
        logger.opt(depth=1).error(message)
