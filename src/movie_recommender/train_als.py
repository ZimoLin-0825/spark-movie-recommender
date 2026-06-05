import argparse
import json
from pathlib import Path

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, collect_set, explode

from movie_recommender.artifact import recommend_from_artifact, write_recommendations_csv
from movie_recommender.paths import (
    MODELS_DIR,
    MOVIELENS_SMALL_DIR,
    MOVIES_CSV,
    RATINGS_CSV,
    REPORTS_DIR,
    ensure_project_dirs,
)
from movie_recommender.spark_utils import build_spark


def load_ratings(spark: SparkSession, ratings_path: Path = RATINGS_CSV) -> DataFrame:
    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing ratings file: {ratings_path}. Run download_data first.")

    return (
        spark.read.option("header", True)
        .csv(str(ratings_path))
        .select(
            col("userId").cast("int").alias("userId"),
            col("movieId").cast("int").alias("movieId"),
            col("rating").cast("float").alias("rating"),
            col("timestamp").cast("long").alias("timestamp"),
        )
        .dropna(subset=["userId", "movieId", "rating"])
    )


def load_movies(spark: SparkSession, movies_path: Path = MOVIES_CSV) -> DataFrame:
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing movies file: {movies_path}. Run download_data first.")

    return (
        spark.read.option("header", True)
        .csv(str(movies_path))
        .select(
            col("movieId").cast("int").alias("movieId"),
            col("title"),
            col("genres"),
        )
        .dropna(subset=["movieId"])
    )


def ranking_metrics_at_k(model, test: DataFrame, k: int) -> dict:
    relevant = test.filter(col("rating") >= 4.0).select("userId", "movieId").distinct()
    users = relevant.select("userId").distinct()

    relevant_count = relevant.count()
    user_count = users.count()
    if relevant_count == 0 or user_count == 0:
        return {
            f"precision_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            "users_evaluated": 0,
            "relevant_test_items": 0,
        }

    recommendations = model.recommendForUserSubset(users, k)
    exploded = (
        recommendations.select("userId", explode("recommendations").alias("rec"))
        .select("userId", col("rec.movieId").alias("movieId"))
        .distinct()
    )
    hits = exploded.join(relevant, ["userId", "movieId"], "inner").count()

    return {
        f"precision_at_{k}": hits / float(user_count * k),
        f"recall_at_{k}": hits / float(relevant_count),
        "users_evaluated": user_count,
        "relevant_test_items": relevant_count,
        "hits_at_k": hits,
    }


def save_factor_artifact(model, ratings: DataFrame, movies: DataFrame, artifact_path: Path) -> dict:
    movie_rows = movies.collect()
    seen_rows = ratings.groupBy("userId").agg(collect_set("movieId").alias("movieIds")).collect()

    artifact = {
        "model_type": "spark_als_factor_artifact",
        "rank": len(model.userFactors.first()["features"]),
        "user_factors": {
            str(row["id"]): [float(value) for value in row["features"]]
            for row in model.userFactors.collect()
        },
        "item_factors": {
            str(row["id"]): [float(value) for value in row["features"]]
            for row in model.itemFactors.collect()
        },
        "movies": {
            str(row["movieId"]): {"title": row["title"], "genres": row["genres"]}
            for row in movie_rows
        },
        "seen_movies": {
            str(row["userId"]): [int(movie_id) for movie_id in row["movieIds"]]
            for row in seen_rows
        },
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def save_sample_recommendations(artifact: dict, user_ids: list[int], top_n: int) -> None:
    rows = []
    for user_id in user_ids:
        rows.extend(recommend_from_artifact(artifact, user_id, top_n))

    output_path = REPORTS_DIR / "sample_recommendations.csv"
    write_recommendations_csv(rows, output_path)
    print(f"Sample recommendations written to {output_path}")


def train(args: argparse.Namespace) -> dict:
    ensure_project_dirs()
    spark = build_spark("spark_movie_recommender_train")
    try:
        data_dir = Path(args.data_dir)
        ratings = load_ratings(spark, data_dir / "ratings.csv")
        movies = load_movies(spark, data_dir / "movies.csv")

        if args.limit_ratings:
            ratings = ratings.orderBy("userId", "movieId").limit(args.limit_ratings)

        train_df, test_df = ratings.randomSplit([0.8, 0.2], seed=args.seed)
        train_df.cache()
        test_df.cache()

        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=args.rank,
            maxIter=args.max_iter,
            regParam=args.reg_param,
            coldStartStrategy="drop",
            nonnegative=True,
            seed=args.seed,
        )
        model = als.fit(train_df)

        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)

        metrics = {
            "ratings_count": ratings.count(),
            "train_count": train_df.count(),
            "test_count": test_df.count(),
            "rank": args.rank,
            "max_iter": args.max_iter,
            "reg_param": args.reg_param,
            "rmse": rmse,
        }
        metrics.update(ranking_metrics_at_k(model, test_df, args.top_k))

        artifact_path = MODELS_DIR / "als_factors.json"
        artifact = save_factor_artifact(model, ratings, movies, artifact_path)
        print(f"Model artifact written to {artifact_path}")

        metrics_path = REPORTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics written to {metrics_path}")
        print(json.dumps(metrics, indent=2))

        user_ids = [
            row["userId"]
            for row in ratings.select("userId").distinct().orderBy("userId").limit(args.sample_users).collect()
        ]
        save_sample_recommendations(artifact, user_ids, args.top_n)
        return metrics
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Spark ALS movie recommender.")
    parser.add_argument("--data-dir", default=str(MOVIELENS_SMALL_DIR), help="MovieLens data directory.")
    parser.add_argument("--rank", type=int, default=10, help="ALS latent factor rank.")
    parser.add_argument("--max-iter", type=int, default=5, help="ALS training iterations.")
    parser.add_argument("--reg-param", type=float, default=0.1, help="ALS regularization parameter.")
    parser.add_argument("--top-k", type=int, default=10, help="K for ranking metrics.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N sample recommendations to write.")
    parser.add_argument("--sample-users", type=int, default=5, help="Number of users for sample recommendation output.")
    parser.add_argument("--limit-ratings", type=int, default=0, help="Optional row limit for fast smoke tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
