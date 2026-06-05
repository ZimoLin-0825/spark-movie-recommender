from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from movie_recommender.artifact import load_artifact, recommend_from_artifact  # noqa: E402
from movie_recommender.paths import MODELS_DIR, REPORTS_DIR  # noqa: E402


DEFAULT_MODEL_PATH = MODELS_DIR / "als_factors.json"
DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"


@st.cache_data(show_spinner=False)
def cached_artifact(model_path: str) -> dict:
    return load_artifact(Path(model_path))


@st.cache_data(show_spinner=False)
def cached_metrics(metrics_path: str) -> dict:
    path = Path(metrics_path)
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def format_user_option(user_id: int) -> str:
    return f"User {user_id}"


def main() -> None:
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon=":movie_camera:",
        layout="wide",
    )

    st.title("Movie Recommendation System")
    st.caption("PySpark ALS recommender trained on MovieLens latest-small.")

    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        st.error(
            "No trained model artifact found. Run "
            "`python -m movie_recommender.train_als --rank 8 --max-iter 3 --reg-param 0.12` first."
        )
        st.stop()

    artifact = cached_artifact(str(model_path))
    metrics = cached_metrics(str(DEFAULT_METRICS_PATH))

    user_ids = sorted(int(user_id) for user_id in artifact["user_factors"].keys())
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Try It")
        selected_user = st.selectbox(
            "Choose a MovieLens user",
            options=user_ids,
            format_func=format_user_option,
            index=0,
        )
        top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10, step=5)
        show_seen = st.toggle("Show user's watched movie IDs", value=False)

        if metrics:
            st.subheader("Latest Training Metrics")
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
            st.metric("Ratings", f"{metrics.get('ratings_count', 0):,}")
            st.metric("Train / Test", f"{metrics.get('train_count', 0):,} / {metrics.get('test_count', 0):,}")

    rows = recommend_from_artifact(artifact, selected_user, top_n)

    with right:
        st.subheader(f"Top {top_n} Recommendations for User {selected_user}")
        if not rows:
            st.warning("No recommendations available for this user.")
        else:
            st.dataframe(
                [
                    {
                        "Rank": idx,
                        "Movie": row["title"],
                        "Genres": row["genres"],
                        "Score": round(row["score"], 3),
                    }
                    for idx, row in enumerate(rows, start=1)
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("How This Works")
        st.markdown(
            """
            The model learns latent factors for users and movies using Spark MLlib ALS.
            For the selected user, the app scores movies the user has not rated yet,
            then ranks them by predicted preference.
            """
        )

        if show_seen:
            seen = artifact.get("seen_movies", {}).get(str(selected_user), [])
            st.write("Watched/rated movie IDs:")
            st.code(", ".join(str(movie_id) for movie_id in seen[:200]) or "None")


if __name__ == "__main__":
    main()
