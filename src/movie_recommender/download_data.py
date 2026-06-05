import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from movie_recommender.paths import MOVIELENS_SMALL_DIR, RAW_DATA_DIR, ensure_project_dirs


MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_file(url: str, destination: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    print(f"Saved {destination}")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"Extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)


def download_movielens(force: bool = False) -> Path:
    ensure_project_dirs()
    ratings_path = MOVIELENS_SMALL_DIR / "ratings.csv"
    movies_path = MOVIELENS_SMALL_DIR / "movies.csv"

    if ratings_path.exists() and movies_path.exists() and not force:
        print(f"MovieLens data already exists at {MOVIELENS_SMALL_DIR}")
        return MOVIELENS_SMALL_DIR

    if force and MOVIELENS_SMALL_DIR.exists():
        shutil.rmtree(MOVIELENS_SMALL_DIR)

    zip_path = RAW_DATA_DIR / "ml-latest-small.zip"
    download_file(MOVIELENS_SMALL_URL, zip_path)
    extract_zip(zip_path, RAW_DATA_DIR)
    zip_path.unlink(missing_ok=True)

    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError("MovieLens extraction did not produce ratings.csv and movies.csv")

    print(f"MovieLens data ready at {MOVIELENS_SMALL_DIR}")
    return MOVIELENS_SMALL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MovieLens latest-small data.")
    parser.add_argument("--force", action="store_true", help="Redownload and replace existing data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_movielens(force=args.force)


if __name__ == "__main__":
    main()
