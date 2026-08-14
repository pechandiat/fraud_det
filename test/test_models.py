"""Tests for src/models.py"""

import pytest
from lightgbm import LGBMClassifier
from src.models import build_model


def test_build_model_returns_correct_class():
    """Test that build_model returns a LGBMClassifier instance for lightgbm model."""
    model, params = build_model("lightgbm")
    assert isinstance(model, LGBMClassifier)


def test_build_model_applies_default_params():
    """Test that build_model applies default parameters correctly."""
    _, params = build_model("random_forest")
    assert params["class_weight"] == "balanced"


def test_build_model_overrides_defaults():
    """Test that build_model overrides default parameters with provided values."""
    _, params = build_model("lightgbm", n_estimators=200)
    assert params["n_estimators"] == 200


def test_build_model_raises_on_unknown_model():
    """Test that build_model raises ValueError for unknown model types."""
    with pytest.raises(ValueError):
        build_model("not_a_real_model")
