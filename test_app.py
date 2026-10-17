from pathlib import Path

import pytest

from inference import load_artifacts, predict_image

SAMPLES = Path(__file__).parent / "samples"


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


@pytest.mark.parametrize("label", ["paper", "rock", "scissors"])
def test_predict_sample(artifacts, label):
    result = predict_image(SAMPLES / f"{label}.png", *artifacts)
    assert result["label"] == label
    assert sorted(i["label"] for i in result["top_k"]) == ["paper", "rock", "scissors"]
    assert abs(sum(i["confidence"] for i in result["top_k"]) - 1) < 1e-3


def test_app_runs():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent / "app.py")).run(timeout=60)
    assert not at.exception
