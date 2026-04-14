from __future__ import annotations

# This file contains small utility helpers that are reused across the project.
# The purpose is to avoid repeating the same setup code in every script.
#
# In this project, the utilities are mainly responsible for:
# - loading the YAML configuration file
# - loading environment variables from .env
# - setting up logging
# - creating required directories
# - saving JSON artifacts
#
# This helps keep the pipeline organized, reproducible, and easier to maintain.

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """
    Load the central YAML configuration file.

    Why this function exists:
    - The project uses one shared config file for paths, model settings,
      API settings, thresholds, and feature definitions.
    - Instead of hardcoding values in multiple scripts, each script can
      call this helper and read from the same configuration source.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The configuration file loaded as a Python dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_env() -> None:
    """
    Load environment variables from a .env file if one exists.

    Why this function exists:
    - Environment variables are useful for values that should not be
      hardcoded in source code, such as API keys or log settings.
    - After calling this function, the rest of the project can access
      those values through os.getenv().

    Returns
    -------
    None
    """
    load_dotenv()


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the project.

    Why this function exists:
    - Logging is important in an MLOps pipeline because it documents what
      happened during ingestion, preprocessing, training, and prediction.
    - A consistent logging format makes debugging and monitoring easier.

    Parameters
    ----------
    level : str
        Logging level as text, for example "INFO", "WARNING", or "DEBUG".

    Returns
    -------
    None
    """
    logging.basicConfig(
        # Convert the string log level into a logging constant.
        # If the provided value is invalid, default to INFO.
        level=getattr(logging, level.upper(), logging.INFO),

        # Standardized log format used across all project scripts.
        # Example output:
        # 2026-04-13 14:44:11,871 | INFO | src.train | Saved model to ...
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(paths: dict[str, str]) -> None:
    """
    Create all configured directories if they do not already exist.

    Why this function exists:
    - The pipeline writes raw data, processed data, models, metrics,
      and predictions to several folders.
    - This helper ensures the folder structure exists before files are saved,
      so the scripts do not fail because of missing directories.

    Parameters
    ----------
    paths : dict[str, str]
        Dictionary containing directory paths, typically loaded from config.

    Returns
    -------
    None
    """
    for path in paths.values():
        # parents=True allows nested folders to be created automatically.
        # exist_ok=True prevents an error if the directory already exists.
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], output_path: str) -> None:
    """
    Save a Python dictionary as a JSON file.

    Why this function exists:
    - The project stores metrics and other artifacts in JSON format.
    - JSON is lightweight, easy to inspect manually, and easy to reuse later.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary to save.
    output_path : str
        File path where the JSON file should be written.

    Returns
    -------
    None
    """
    # Make sure the parent directory exists before writing the file.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # indent=2 makes the JSON human-readable,
        # which is useful for debugging and reporting.
        json.dump(data, f, indent=2)


def get_log_level() -> str:
    """
    Read the logging level from environment variables.

    Why this function exists:
    - It allows log verbosity to be controlled externally through .env
      instead of editing Python code.
    - If LOG_LEVEL is not set, the project defaults to INFO.

    Returns
    -------
    str
        The selected log level.
    """
    return os.getenv("LOG_LEVEL", "INFO")