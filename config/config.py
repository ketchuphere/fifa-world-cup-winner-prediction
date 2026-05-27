"""
config/config.py
----------------
Loads config.yml and exposes a single `CFG` dict throughout the project.
All paths are resolved relative to the project root (one level above /config).
"""

from pathlib import Path
import yaml

_CONFIG_FILE = Path(__file__).parent / "config.yml"
_PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: Path = _CONFIG_FILE) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve all path entries to absolute paths
    for key, rel_path in cfg["paths"].items():
        cfg["paths"][key] = str(_PROJECT_ROOT / rel_path)

    cfg["project_root"] = str(_PROJECT_ROOT)
    return cfg


CFG = load_config()


#Convenience accessors

def raw_path(filename: str) -> Path:
    return Path(CFG["paths"]["raw"]) / filename

def preprocessed_path(filename: str) -> Path:
    return Path(CFG["paths"]["preprocessed"]) / filename

def features_path(filename: str) -> Path:
    return Path(CFG["paths"]["features"]) / filename

def predictions_path(filename: str) -> Path:
    return Path(CFG["paths"]["predictions"]) / filename
