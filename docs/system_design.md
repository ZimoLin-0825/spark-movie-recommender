# Recommendation System Design

This project implements a compact version of a two-stage recommendation system.

## Problem

Recommend movies that a user is likely to enjoy based on historical ratings.

Business-style goal:

- Increase user engagement by recommending relevant movies.
- Keep recommendations explainable enough for a portfolio project.

Technical goal:

- Learn user and movie latent factors from sparse interaction data.
- Rank unseen movies for each user.

## Data Flow

```text
MovieLens ratings.csv
        |
        v
Spark DataFrame cleaning and train/test split
        |
        v
ALS model training
        |
        +--> Offline evaluation: RMSE, Precision@K, Recall@K
        |
        v
Top-N recommendation generation
        |
        v
CSV reports and CLI output
```

## Candidate Generation and Ranking

The ALS model plays both roles in this small project:

- Candidate generation: produce a shortlist of likely movies for a user.
- Ranking: sort candidates by predicted preference score.

In a production-scale system, candidate generation might use approximate
nearest neighbors or cached generators, while a separate ranking model reranks
the candidate set with more features. This project keeps the scope practical
for one person while preserving the same system-design logic.

## Features

Current model:

- `userId`
- `movieId`
- `rating`

Possible extensions:

- Movie genres from `movies.csv`
- Recency weighting from `timestamp`
- Popularity-based cold-start fallback
- Two-tower neural recommender using user and item features

## Evaluation

Offline metrics:

- RMSE for rating prediction quality
- Precision@K and Recall@K for recommendation quality

Online metrics in a real product would include:

- Watch time
- Click-through rate
- Conversion rate
- Session length
- Retention

## Limitations

- MovieLens ratings are explicit feedback, not real streaming behavior.
- ALS has limited cold-start ability for unseen users and movies.
- The project does not include online A/B testing.

These limitations are acceptable for a DS practical, and they are useful to
discuss in the README or presentation.
