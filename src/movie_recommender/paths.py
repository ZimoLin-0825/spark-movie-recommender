from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MOVIELENS_SMALL_DIR = RAW_DATA_DIR / "ml-latest-small"
RATINGS_CSV = MOVIELENS_SMALL_DIR / "ratings.csv"
MOVIES_CSV = MOVIELENS_SMALL_DIR / "movies.csv"


def ensure_project_dirs() -> None:
    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
