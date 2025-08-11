import os
import sys
from loguru import logger

logger.disable("sds") # By default, disable all loggers whose names start with "sds".
env_log_level = os.getenv("SDS_LOG_LEVEL", "INFO") # Check for the user-defined environment variable.

if env_log_level:
    try:
        logger.remove(0)
    except ValueError:
        pass

    log_level = env_log_level.upper()
    logger.add(sys.stderr, level=log_level)
    logger.enable("sds")
    logger.debug(f"Logging for 'sds' and its submodules enabled at '{log_level}' level.")
