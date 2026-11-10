import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def create_run_dir(config, root: str = "runs") -> Path:
    """runs/<dataset>/<date>_g<gamma>_s<seed>/ with the full config saved inside"""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(root) / config.data.name / f"{stamp}_g{config.training.gamma}_s{config.seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=4))
    return run_dir


class CSVLogger:
    """Appends one row per call and closes the file right away: a crash keeps every finished epoch"""

    def __init__(self, path: Path):
        self.path = path
        self.columns = None

    def log(self, row: dict):
        if self.columns is None:                    # the first row fixes the columns
            self.columns = list(row)
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.columns).writeheader()
        if list(row) != self.columns:
            raise ValueError(f"{self.path.name}: the keys of this row differ from the header")
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(row)