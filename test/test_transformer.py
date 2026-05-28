"""Tests for RowTransformer and PatchTransformer on MNIST."""

import numpy as np
import pytest

from beyond_backprop.constants import DATA_PATH
from beyond_backprop.mnist import Split, load_mnist
from beyond_backprop.transformer import PatchTransformer, RowTransformer

IMAGE_H = 28
IMAGE_W = 28
N_CLASSES = 10
D_MODEL = 32   # small model for fast tests
N_HEADS = 4
N_BLOCKS = 1


# ── fixtures ───────────────────────────────────────────────────────────────────


def make_row_transformer(seed: int = 0) -> RowTransformer:
    """Create a small RowTransformer for testing."""
    return RowTransformer(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        d_ff=D_MODEL * 4,
        n_classes=N_CLASSES,
        seed=seed,
    )


def make_patch_transformer(patch_size: int = 4, seed: int = 0) -> PatchTransformer:
    """Create a small PatchTransformer for testing."""
    return PatchTransformer(
        patch_size=patch_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        d_ff=D_MODEL * 4,
        n_classes=N_CLASSES,
        seed=seed,
    )


@pytest.fixture(scope="module")
def mnist_train() -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 28, 28) float32 images in [0,1] and int64 labels."""
    images, labels = load_mnist(DATA_PATH, Split.TRAIN)
    return images.astype(np.float32) / 255.0, labels.astype(np.int64)


@pytest.fixture(scope="module")
def mnist_test() -> tuple[np.ndarray, np.ndarray]:
    """Return (N, 28, 28) float32 test images in [0,1] and int64 labels."""
    images, labels = load_mnist(DATA_PATH, Split.TEST)
    return images.astype(np.float32) / 255.0, labels.astype(np.int64)


@pytest.fixture(scope="module")
def fitted_row(mnist_train: tuple[np.ndarray, np.ndarray]) -> RowTransformer:
    """Train a small RowTransformer for reuse across tests."""
    X, y = mnist_train
    model = make_row_transformer(seed=0)
    model.fit(X, y, epochs=2, batch_size=512)
    return model


@pytest.fixture(scope="module")
def fitted_patch(mnist_train: tuple[np.ndarray, np.ndarray]) -> PatchTransformer:
    """Train a small PatchTransformer for reuse across tests."""
    X, y = mnist_train
    model = make_patch_transformer(seed=0)
    model.fit(X, y, epochs=2, batch_size=512)
    return model


# ── construction ───────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_transformer_construction() -> None:
    """RowTransformer constructs without error."""
    model = make_row_transformer()
    assert model is not None


@pytest.mark.skip(reason="not implemented yet")
def test_patch_transformer_construction_default() -> None:
    """PatchTransformer with default patch_size=4 constructs without error."""
    model = make_patch_transformer(patch_size=4)
    assert model is not None


@pytest.mark.skip(reason="not implemented yet")
def test_patch_transformer_construction_patch7() -> None:
    """PatchTransformer with patch_size=7 (→ 16 tokens of dim 49) constructs."""
    model = make_patch_transformer(patch_size=7)
    assert model is not None


@pytest.mark.skip(reason="not implemented yet")
def test_patch_transformer_invalid_patch_size() -> None:
    """PatchTransformer raises ValueError when patch_size does not divide 28."""
    with pytest.raises(ValueError):
        PatchTransformer(patch_size=5)


# ── forward pass shapes ────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_transformer_forward_shape(
    fitted_row: RowTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """predict() returns (N,) int64 array for RowTransformer."""
    X, _ = mnist_test
    preds = fitted_row.predict(X[:32])
    assert preds.shape == (32,)
    assert preds.dtype == np.int64


@pytest.mark.skip(reason="not implemented yet")
def test_patch_transformer_forward_shape(
    fitted_patch: PatchTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """predict() returns (N,) int64 array for PatchTransformer."""
    X, _ = mnist_test
    preds = fitted_patch.predict(X[:32])
    assert preds.shape == (32,)
    assert preds.dtype == np.int64


# ── predict output values ──────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_predictions_in_range(
    fitted_row: RowTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """RowTransformer predictions are in [0, 9]."""
    X, _ = mnist_test
    preds = fitted_row.predict(X[:100])
    assert int(preds.min()) >= 0
    assert int(preds.max()) <= 9


@pytest.mark.skip(reason="not implemented yet")
def test_patch_predictions_in_range(
    fitted_patch: PatchTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """PatchTransformer predictions are in [0, 9]."""
    X, _ = mnist_test
    preds = fitted_patch.predict(X[:100])
    assert int(preds.min()) >= 0
    assert int(preds.max()) <= 9


# ── accuracy ───────────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_accuracy_range(
    fitted_row: RowTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """accuracy() returns a plain float in [0, 1]."""
    X, y = mnist_test
    acc = fitted_row.accuracy(X, y)
    assert type(acc) is float
    assert 0.0 <= acc <= 1.0


@pytest.mark.skip(reason="not implemented yet")
def test_patch_accuracy_range(
    fitted_patch: PatchTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """accuracy() returns a plain float in [0, 1]."""
    X, y = mnist_test
    acc = fitted_patch.accuracy(X, y)
    assert type(acc) is float
    assert 0.0 <= acc <= 1.0


# ── fit returns self ───────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_fit_returns_self(
    mnist_train: tuple[np.ndarray, np.ndarray],
) -> None:
    """fit() returns the model instance."""
    X, y = mnist_train
    model = make_row_transformer()
    result = model.fit(X[:500], y[:500], epochs=1)
    assert result is model


@pytest.mark.skip(reason="not implemented yet")
def test_patch_fit_returns_self(
    mnist_train: tuple[np.ndarray, np.ndarray],
) -> None:
    """fit() returns the model instance."""
    X, y = mnist_train
    model = make_patch_transformer()
    result = model.fit(X[:500], y[:500], epochs=1)
    assert result is model


# ── reproducibility ────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_same_seed_same_predictions(
    mnist_train: tuple[np.ndarray, np.ndarray],
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """Two RowTransformers with the same seed produce identical predictions."""
    X_tr, y_tr = mnist_train
    X_te, _ = mnist_test
    m1 = make_row_transformer(seed=7)
    m1.fit(X_tr[:1000], y_tr[:1000], epochs=1)
    m2 = make_row_transformer(seed=7)
    m2.fit(X_tr[:1000], y_tr[:1000], epochs=1)
    np.testing.assert_array_equal(m1.predict(X_te[:50]), m2.predict(X_te[:50]))


# ── attention maps ─────────────────────────────────────────────────────────────


@pytest.mark.skip(reason="not implemented yet")
def test_row_attention_maps_shape(
    fitted_row: RowTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """attention_maps() returns n_blocks arrays of shape (B, T+1, T+1) for RowTransformer.

    T=28 row tokens + 1 CLS token → T+1=29.
    """
    X, _ = mnist_test
    maps = fitted_row.attention_maps(X[:8])
    assert len(maps) == N_BLOCKS
    assert maps[0].shape == (8, 29, 29)


@pytest.mark.skip(reason="not implemented yet")
def test_patch_attention_maps_shape(
    fitted_patch: PatchTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """attention_maps() returns n_blocks arrays of shape (B, T+1, T+1) for PatchTransformer.

    patch_size=4 → T=49 patch tokens + 1 CLS token → T+1=50.
    """
    X, _ = mnist_test
    maps = fitted_patch.attention_maps(X[:8])
    assert len(maps) == N_BLOCKS
    assert maps[0].shape == (8, 50, 50)


@pytest.mark.skip(reason="not implemented yet")
def test_attention_weights_sum_to_one(
    fitted_row: RowTransformer,
    mnist_test: tuple[np.ndarray, np.ndarray],
) -> None:
    """Attention weight rows sum to 1.0 (valid probability distributions)."""
    X, _ = mnist_test
    maps = fitted_row.attention_maps(X[:4])
    for attn in maps:
        row_sums = attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
