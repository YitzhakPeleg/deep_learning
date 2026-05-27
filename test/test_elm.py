"""Tests for the Extreme Learning Machine (ELM) implementation."""

import numpy as np
import pytest

from beyond_backprop.constants import DATA_PATH
from beyond_backprop.elm import ELM, Activation
from beyond_backprop.mnist import Split, load_mnist

HIDDEN_SIZE = 20
N_CLASSES = 10
INPUT_SHAPE = (28, 28)
INPUT_SIZE = int(np.prod(INPUT_SHAPE))


# ── fixtures ──────────────────────────────────────────────────────────────────
def create_elm(hidden_size: int, seed: int) -> ELM:
    """Helper to create an ELM with consistent hyperparameters."""
    return ELM(
        input_size=INPUT_SIZE,
        output_size=N_CLASSES,
        activation=Activation.SIGMOID,
        regularization_factor=1e-4,
        hidden_size=hidden_size,
        seed=seed,
    )


@pytest.fixture(scope="module")
def mnist_train() -> tuple[np.ndarray, np.ndarray]:
    """Return flattened float64 training images and integer labels."""
    images, labels = load_mnist(DATA_PATH, Split.TRAIN)
    X = images.reshape(len(images), -1).astype(np.float64) / 255.0
    return X, labels.astype(np.int64)


@pytest.fixture(scope="module")
def mnist_test() -> tuple[np.ndarray, np.ndarray]:
    """Return flattened float64 test images and integer labels."""
    images, labels = load_mnist(DATA_PATH, Split.TEST)
    X = images.reshape(len(images), -1).astype(np.float64) / 255.0
    return X, labels.astype(np.int64)


@pytest.fixture(scope="module")
def fitted_elm(mnist_train: tuple[np.ndarray, np.ndarray]) -> ELM:
    """Train a small ELM once and reuse it across tests."""
    X, y = mnist_train
    elm = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    elm.fit(X, y)
    return elm


# ── construction & attributes ─────────────────────────────────────────────────


def test_attributes_none_before_fit() -> None:
    """W1, b, and beta are None before fit is called."""
    elm = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    assert elm.W1 is None
    assert elm.b is None
    assert elm.beta is None
    assert elm.n_classes == N_CLASSES


def test_attributes_set_after_fit(fitted_elm: ELM) -> None:
    """W1, b, beta, and n_classes are populated after fit."""
    assert fitted_elm.W1 is not None
    assert fitted_elm.b is not None
    assert fitted_elm.beta is not None
    assert fitted_elm.n_classes == N_CLASSES


# ── weight shapes ─────────────────────────────────────────────────────────────


def test_W1_shape(fitted_elm: ELM) -> None:
    """W1 has shape (784, hidden_size)."""
    assert fitted_elm.W1.shape == (INPUT_SIZE, HIDDEN_SIZE)


def test_b_shape(fitted_elm: ELM) -> None:
    """b has shape (hidden_size,)."""
    assert fitted_elm.b.shape == (HIDDEN_SIZE,)


def test_beta_shape(fitted_elm: ELM) -> None:
    """beta has shape (hidden_size, n_classes)."""
    assert fitted_elm.beta.shape == (HIDDEN_SIZE, N_CLASSES)


# ── predict output shapes and dtypes ─────────────────────────────────────────


def test_predict_proba_shape(
    fitted_elm: ELM, mnist_test: tuple[np.ndarray, np.ndarray]
) -> None:
    """predict_proba returns (N, 10) float64 array."""
    X, _ = mnist_test
    proba = fitted_elm.predict_proba(X)
    assert proba.shape == (len(X), N_CLASSES)
    assert proba.dtype == np.float64


def test_predict_shape_and_dtype(
    fitted_elm: ELM, mnist_test: tuple[np.ndarray, np.ndarray]
) -> None:
    """predict returns (N,) int64 array with values in [0, 9]."""
    X, _ = mnist_test
    preds = fitted_elm.predict(X)
    assert preds.shape == (len(X),)
    assert preds.dtype == np.int64
    assert int(preds.min()) >= 0
    assert int(preds.max()) <= 9


# ── reproducibility ───────────────────────────────────────────────────────────


def test_same_seed_same_weights(mnist_train: tuple[np.ndarray, np.ndarray]) -> None:
    """Two ELMs with the same seed produce identical W1 and b."""
    X, y = mnist_train
    elm_a = create_elm(hidden_size=HIDDEN_SIZE, seed=42)
    elm_a.fit(X, y)
    elm_b = create_elm(hidden_size=HIDDEN_SIZE, seed=42)
    elm_b.fit(X, y)
    np.testing.assert_array_equal(elm_a.W1, elm_b.W1)
    np.testing.assert_array_equal(elm_a.b, elm_b.b)


def test_different_seeds_different_weights(
    mnist_train: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two ELMs with different seeds produce different W1."""
    X, y = mnist_train
    elm_a = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    elm_a.fit(X, y)
    elm_b = create_elm(hidden_size=HIDDEN_SIZE, seed=1)
    elm_b.fit(X, y)
    assert not np.array_equal(elm_a.W1, elm_b.W1)


# ── fit returns self ──────────────────────────────────────────────────────────


def test_fit_returns_self(mnist_train: tuple[np.ndarray, np.ndarray]) -> None:
    """fit() returns the ELM instance (enables chaining)."""
    X, y = mnist_train
    elm = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    result = elm.fit(X, y)
    assert result is elm


# ── before-fit guards ─────────────────────────────────────────────────────────


def test_predict_before_fit_raises() -> None:
    """predict() raises RuntimeError if called before fit."""
    elm = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    X = np.zeros((23, INPUT_SIZE))
    with pytest.raises(RuntimeError):
        elm.predict(X)


def test_predict_proba_before_fit_raises() -> None:
    """predict_proba() raises RuntimeError if called before fit."""
    elm = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    X = np.zeros((23, INPUT_SIZE))
    with pytest.raises(RuntimeError):
        elm.predict_proba(X)


# ── accuracy ──────────────────────────────────────────────────────────────────


def test_accuracy_range(
    fitted_elm: ELM, mnist_test: tuple[np.ndarray, np.ndarray]
) -> None:
    """accuracy() returns a plain float in [0, 1]."""
    X, y = mnist_test
    acc = fitted_elm.accuracy(X, y)
    assert type(acc) is float
    assert 0.0 <= acc <= 1.0


# ── activations ───────────────────────────────────────────────────────────────


def test_tanh_activation(mnist_train: tuple[np.ndarray, np.ndarray]) -> None:
    """ELM with tanh activation trains and predicts without error."""
    X, y = mnist_train
    elm = ELM(
        input_size=784,
        hidden_size=200,
        output_size=10,
        activation=Activation.TANH,
        regularization_factor=1e-4,
        seed=0,
    )
    elm.fit(X, y)
    preds = elm.predict(X[:100])
    assert preds.shape == (100,)


# ── regularization ────────────────────────────────────────────────────────────


def test_regularization_changes_beta(
    mnist_train: tuple[np.ndarray, np.ndarray],
) -> None:
    """Different regularization_factor values produce different output weights."""
    X, y = mnist_train
    elm_low = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    elm_low._regularization_factor = 1e-4
    elm_low.fit(X, y)
    elm_high = create_elm(hidden_size=HIDDEN_SIZE, seed=0)
    elm_high._regularization_factor = 1
    elm_high.fit(X, y)
    assert not np.allclose(elm_low.beta, elm_high.beta)
