from pathlib import Path

import numpy as np
import pytest

from beyond_backprop.mnist import Split, load_mnist

DATA_DIR = Path(__file__).parent.parent / "data"


def test_train_shapes() -> None:
    """Train split returns 60 000 images of shape (28, 28) and matching labels."""
    images, labels = load_mnist(DATA_DIR, Split.TRAIN)
    assert images.shape == (60_000, 28, 28)
    assert labels.shape == (60_000,)


def test_test_shapes() -> None:
    """Test split returns 10 000 images of shape (28, 28) and matching labels."""
    images, labels = load_mnist(DATA_DIR, Split.TEST)
    assert images.shape == (10_000, 28, 28)
    assert labels.shape == (10_000,)


def test_pixel_range() -> None:
    """All pixel values lie in [0, 255] and dtype is uint8."""
    images, _ = load_mnist(DATA_DIR, Split.TRAIN)
    assert images.dtype == np.uint8
    assert int(images.min()) >= 0
    assert int(images.max()) <= 255


def test_label_range() -> None:
    """All labels lie in [0, 9] and dtype is uint8."""
    _, labels = load_mnist(DATA_DIR, Split.TRAIN)
    assert labels.dtype == np.uint8
    assert int(labels.min()) >= 0
    assert int(labels.max()) <= 9


def test_invalid_split() -> None:
    """An unrecognised split string raises ValueError."""
    with pytest.raises(ValueError):
        load_mnist(DATA_DIR, "val")


def test_str_split() -> None:
    """Plain strings are coerced to Split — no Split import required at call site."""
    images, labels = load_mnist(DATA_DIR, "test")
    assert images.shape == (10_000, 28, 28)
    assert labels.shape == (10_000,)
