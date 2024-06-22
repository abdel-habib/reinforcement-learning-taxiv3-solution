from loguru import logger
import sys

# Custom log format
fmt = "{message}"
config = {
    "handlers": [
        {"sink": sys.stderr, "format": fmt},
    ],
}
logger.configure(**config)


def testlog():
    logger.info("Hello, world!")