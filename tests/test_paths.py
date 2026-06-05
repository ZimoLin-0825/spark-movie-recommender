from movie_recommender.paths import PROJECT_ROOT


def test_project_root_points_to_repo() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").exists()
