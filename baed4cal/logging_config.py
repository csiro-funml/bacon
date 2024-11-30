import logging.config


DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "standard": {"format": "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"},
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {"": {"handlers": ["console"], "level": "INFO", "propagate": True}},
    "disable_existing_loggers": False,
}

logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


def make_file_config(
    filename: str,
    logger_name: str = "",
    include_warnings: bool = False,
    handler_name: str = "file",
    propagate: bool = True,
    logger_level: str = "DEBUG",
):
    log_config = DEFAULT_LOGGING_CONFIG.copy()
    log_config["handlers"][handler_name] = {
        "level": logger_level,
        "class": "logging.FileHandler",
        "formatter": "standard",
        "filename": filename,
    }
    if logger_name:
        log_config["loggers"][logger_name] = {
            "handlers": [handler_name],
            "level": logger_level,
            "propagate": propagate,
        }
    else:
        log_config["loggers"][""]["handlers"].append(handler_name)

    if include_warnings:
        log_config["loggers"]["py.warnings"] = {
            "handlers": [handler_name],
            "propagate": False,
        }
    return log_config


def make_file_logger(
    filename: str,
    logger_name: str = "",
    include_warnings: bool = False,
    handler_name: str = "file",
    propagate: bool = True,
    logger_level: str = "DEBUG",
):
    log_config = make_file_config(
        filename, logger_name, include_warnings, handler_name, propagate, logger_level
    )
    logging.config.dictConfig(log_config)
    logging.captureWarnings(include_warnings)
    logger = logging.getLogger(logger_name)
    return logger
