"""Repository locations and the local config/paths.yaml."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
SCENARIOS_DIR = REPO_ROOT / "scenarios"
INVENTORY_CSV = REPO_ROOT / "data" / "fault_inventory.csv"
PATHS_FILE = REPO_ROOT / "config" / "paths.yaml"


def load_paths(path=PATHS_FILE) -> dict[str, Path]:
    """DATA_DIR, REPORTS_DIR, WORK_DIR from config/paths.yaml (empty if the file is absent)."""
    path = Path(path)
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    return {k: Path(v).expanduser() for k, v in raw.items() if v}


def work_dir(scenario: str) -> Path:
    paths = load_paths()
    if "WORK_DIR" not in paths:
        raise FileNotFoundError(f"WORK_DIR not set in {PATHS_FILE}, pass --outdir")
    return paths["WORK_DIR"] / scenario
