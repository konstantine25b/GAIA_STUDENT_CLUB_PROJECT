"""Project paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "initial"
DATASET_CSV = DATA_DIR / "mmlu_pro_sample_20_per_category.csv"
MODELS_FILE = ROOT / "config" / "models.txt"
INFERENCE_CONFIG = ROOT / "config" / "inference.json"
RESULTS_DIR = ROOT / "results" / "runs"
