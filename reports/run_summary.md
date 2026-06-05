# Run Summary

Generated from the first local smoke run on MovieLens `ml-latest-small`.

## Training Setup

- Model: Spark MLlib ALS
- Ratings: 100,836
- Train rows: 80,578
- Test rows: 20,258
- Rank: 8
- Max iterations: 3
- Regularization: 0.12

## Metrics

- RMSE: 0.8818
- Precision@10: 0.00017
- Recall@10: 0.00010

The RMSE result is the most useful metric for this baseline run. The top-K
ranking metrics are very sparse because the held-out relevant movies are a tiny
subset of the full movie catalog.

## Example Output

The CLI successfully generated top-N recommendations:

```powershell
python -m movie_recommender.recommend --user-id 1 --top-n 10
```

Sample first recommendation:

```text
Dragon Ball Z: The History of Trunks (1993) | genres=Action|Adventure|Animation
```
