import argparse
import sys
from pathlib import Path

from movie_recommender.artifact import load_artifact, recommend_from_artifact
from movie_recommender.paths import MODELS_DIR


def recommend_for_user(user_id: int, top_n: int, model_path: Path) -> list[dict]:
    artifact = load_artifact(model_path)
    return recommend_from_artifact(artifact, user_id, top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-N movie recommendations for one user.")
    parser.add_argument("--user-id", type=int, required=True, help="Existing MovieLens user ID.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of recommendations to print.")
    parser.add_argument(
        "--model-path",
        default=str(MODELS_DIR / "als_factors.json"),
        help="Path to the saved ALS factor artifact.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    args = parse_args()
    rows = recommend_for_user(args.user_id, args.top_n, Path(args.model_path))
    if not rows:
        print(f"No recommendations found for user {args.user_id}. Try a user seen during training.")
        return

    print(f"Top {len(rows)} recommendations for user {args.user_id}:")
    for idx, row in enumerate(rows, start=1):
        print(f"{idx:02d}. {row['title']} | score={row['score']:.3f} | genres={row['genres']}")


if __name__ == "__main__":
    main()
