import csv
import json
from pathlib import Path

import numpy as np


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run train_als first.")
    return json.loads(path.read_text(encoding="utf-8"))


def recommend_from_artifact(artifact: dict, user_id: int, top_n: int) -> list[dict]:
    user_key = str(user_id)
    user_factors = artifact["user_factors"]
    item_factors = artifact["item_factors"]
    movies = artifact["movies"]
    seen = set(artifact.get("seen_movies", {}).get(user_key, []))

    if user_key not in user_factors:
        return []

    user_vector = np.array(user_factors[user_key], dtype=float)
    scored = []
    for movie_key, factors in item_factors.items():
        movie_id = int(movie_key)
        if movie_id in seen:
            continue
        score = float(np.dot(user_vector, np.array(factors, dtype=float)))
        movie = movies.get(movie_key, {})
        scored.append(
            {
                "userId": int(user_id),
                "movieId": movie_id,
                "score": score,
                "title": movie.get("title", ""),
                "genres": movie.get("genres", ""),
            }
        )

    return sorted(scored, key=lambda row: row["score"], reverse=True)[:top_n]


def write_recommendations_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["userId", "movieId", "score", "title", "genres"])
        writer.writeheader()
        writer.writerows(rows)
