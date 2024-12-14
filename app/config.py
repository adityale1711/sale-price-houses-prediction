import sys
import logging

from types import FrameType
from loguru import logger
from typing import List, cast
from pydantic import BaseSettings, AnyHttpUrl


class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO


class Settings(BaseSettings):
    API_V1_STR: str = '/api/v1'

    # Meta
    logging: LoggingSettings = LoggingSettings()

    # BACKEND_CORS_ORIGINS is a comma-separated list of origins
    # e.g: http://localhost,http://localhost:4200,http://localhost:3000 type: ignore
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        'http://localhost:3000',
        'http://localhost:8000',
        'https://localhost:3000',
        'https://localhost:8000'
    ]

    PROJECT_NAME: str = 'House Price Prediction API'

    class config:
        case_sensitive = True


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # find caller from where oriented the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage()
        )


# Prepare custom logging
def setup_app_logging(config: Settings) -> None:
    LOGGERS = ('uvicorn.asgi', 'uvicorn.access')
    logging.getLogger().handlers = [InterceptHandler()]

    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{'sink': sys.stderr, 'level':config.logging.LOGGING_LEVEL}]
    )


settings = Settings()