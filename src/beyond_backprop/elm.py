"""Extreme Learning Machine (ELM) for MNIST classification.

Architecture
------------
Input → random frozen hidden layer → analytic output weights (ridge regression).

Reference: Huang et al. (2006), "Extreme learning machine: Theory and
applications", Neurocomputing.
"""
# %%

import sys
from enum import Enum
from typing import Self

import numpy as np
import torch
from loguru import logger

logger.remove()  # remove the default stderr sink
logger.add(sys.stderr, level="INFO")


class Activation(Enum):
    SIGMOID = torch.nn.Sigmoid()
    TANH = torch.nn.Tanh()
    RELU = torch.nn.ReLU()


class ELM:
    """Single-hidden-layer feedforward network trained by ridge regression.

    Only the output weights ``beta`` are learned; the hidden-layer weights
    ``W1`` and biases ``b`` are sampled once at construction and never updated.

    Parameters
    ----------
    input_size : int
        Number of input features (784 for flattened MNIST).
    hidden_size : int
        Number of hidden neurons (``L`` in the doc).  More neurons → better
        representation up to the point where regularization is needed.
    output_size : int
        Number of output classes (10 for MNIST).
    activation : Activation
        Hidden-layer activation.  Prefer ``Activation.SIGMOID`` or
        ``Activation.TANH``; ``Activation.RELU`` causes permanently dead
        neurons with random weights (see doc).
    regularization_factor : float
        Ridge regularization coefficient ``λ``.  Added to the diagonal of
        ``HᵀH`` before solving to improve conditioning and reduce overfitting.
    seed : int | None
        Seed for the NumPy RNG used to draw ``W1`` and ``b``.  ``None`` means
        non-reproducible.

    Attributes
    ----------
    W1 : np.ndarray | None
        Hidden-layer weight matrix, shape ``(input_size, hidden_size)``.
        ``None`` before :meth:`fit` is called.
    b : np.ndarray | None
        Hidden-layer bias vector, shape ``(hidden_size,)``.
        ``None`` before :meth:`fit` is called.
    beta : np.ndarray | None
        Output weight matrix, shape ``(hidden_size, output_size)``.
        ``None`` before :meth:`fit` is called.
    n_classes : int
        Number of output classes (equal to ``output_size``).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: Activation,
        regularization_factor: float,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._hidden_layer = torch.nn.Linear(input_size, hidden_size)
        self._hidden_layer.weight.data = torch.from_numpy(
            rng.standard_normal((hidden_size, input_size))
        )
        self._hidden_layer.bias.data = torch.from_numpy(
            rng.standard_normal((hidden_size,))
        )
        self._activation = activation.value
        self._regularization_factor = regularization_factor
        self.n_classes = output_size
        self.output_layer = torch.nn.Linear(hidden_size, self.n_classes, bias=False)

        self.model = torch.nn.Sequential(
            self._hidden_layer, self._activation, self.output_layer
        )
        self._fitted = False

        if activation is Activation.RELU:
            logger.warning(
                "ReLU is not recommended for ELMs: randomly dead neurons are "
                "permanent and degrade representation quality. Prefer SIGMOID or TANH."
            )

    def __repr__(self) -> str:
        return self.model.__repr__()

    @property
    def W1(self) -> np.ndarray | None:
        """Hidden-layer weights, shape ``(input_size, hidden_size)``; ``None`` before fit."""
        if not self._fitted:
            return None
        return self._hidden_layer.weight.data.numpy().T

    @property
    def b(self) -> np.ndarray | None:
        """Hidden-layer biases, shape ``(hidden_size,)``; ``None`` before fit."""
        if not self._fitted:
            return None
        return self._hidden_layer.bias.data.numpy()

    @property
    def beta(self) -> np.ndarray | None:
        """Output weights, shape ``(hidden_size, n_classes)``; ``None`` before fit."""
        if not self._fitted:
            return None
        return self.output_layer.weight.data.numpy().T

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Draw random weights and solve for output weights analytically.

        Draws ``W1`` and ``b`` from a standard normal distribution, computes
        ``H = activation(X W1 + b)``, then solves::

            β = (HᵀH + λI)⁻¹ Hᵀ Y

        where ``Y`` is the one-hot encoding of ``y``.

        Parameters
        ----------
        X : np.ndarray
            Training images, shape ``(N, input_size)``, dtype ``float64``.
        y : np.ndarray
            Integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.

        Returns
        -------
        Self
            ``self``, for chaining.
        """
        logger.debug("running hidden layer forward pass to compute H")
        H: torch.Tensor = self._activation(self._hidden_layer(self._prepare_input(X)))
        logger.debug("converting H to numpy array for SVD")
        H_numpy = H.detach().numpy()
        logger.debug(
            f"computing Moore-Penrose pseudoinverse of H {H_numpy.shape} with "
            f"ridge regularization λ={self._regularization_factor}"
        )
        U, S, Vt = np.linalg.svd(H_numpy, full_matrices=False)
        logger.debug("inverting singular values with ridge regularization")
        S_inv = S / (S**2 + self._regularization_factor)
        logger.debug("computing pseudoinverse of H: H⁺ = V S⁻¹ Uᵀ")
        H_pinv = Vt.T @ np.diag(S_inv) @ U.T
        logger.debug("converting Y to one-hot and solving for β")
        Y = self._one_hot(y)
        beta_T = np.ascontiguousarray((H_pinv @ Y).T)  # (n_classes, hidden_size)
        self.output_layer.weight.data = torch.from_numpy(beta_T)
        self._fitted = True
        logger.info("fit complete")
        return self

    def _prepare_input(self, X: np.ndarray) -> torch.Tensor:
        """Reshape and convert input to a float64 torch.Tensor.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, input_size)`` or ``(N, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, input_size)``, dtype ``float64``.
        """
        return torch.from_numpy(X.reshape(len(X), -1).astype(dtype=np.float64))

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoding.

        Parameters
        ----------
        y : np.ndarray
            Integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.

        Returns
        -------
        np.ndarray
            One-hot encoded labels, shape ``(N, n_classes)``, dtype ``float64``.
        """
        num_samples = y.shape[0]
        one_hot_matrix = np.zeros(shape=(num_samples, self.n_classes), dtype=np.float64)
        one_hot_matrix[np.arange(num_samples), y] = 1.0
        return one_hot_matrix

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ELM is not fitted yet. Call fit() before predicting.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute raw (unnormalized) class scores for each sample.

        Returns ``H β`` without a final softmax so that the caller can apply
        argmax directly for hard predictions or softmax for probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, input_size)``, dtype ``float64``.

        Returns
        -------
        np.ndarray
            Score matrix, shape ``(N, n_classes)``, dtype ``float64``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        self._check_fitted()
        return self.model(self._prepare_input(X)).detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, input_size)``, dtype ``float64``.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(N,)``, dtype ``int64``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute fraction of correctly classified samples.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, input_size)``, dtype ``float64``.
        y : np.ndarray
            True labels, shape ``(N,)``.

        Returns
        -------
        float
            Accuracy in ``[0.0, 1.0]``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        return float(np.mean(self.predict(X) == y))


# %%
if __name__ == "__main__":
    import polars as pl

    from beyond_backprop.constants import DATA_PATH
    from beyond_backprop.mnist import load_mnist

    X_train, y_train = load_mnist(data_dir=DATA_PATH, split="train")
    print(
        f"Loaded MNIST train set: {X_train.shape[0]} samples, each sample with shape {X_train.shape[1:]}"
    )

    X_test, y_test = load_mnist(data_dir=DATA_PATH, split="test")
    print(
        f"Loaded MNIST test set: {X_test.shape[0]} samples, each sample with shape {X_test.shape[1:]}"
    )

    results = []
    for key, activation in Activation.__members__.items():
        for hidden_size in [1, 10, 50, 100, 500, 1000]:
            for seed in range(100):
                elm = ELM(
                    input_size=784,
                    hidden_size=hidden_size,
                    output_size=10,
                    activation=activation,
                    regularization_factor=1e-3,
                    seed=seed,
                ).fit(X_train, y_train)
                train_acc = elm.accuracy(X_train, y_train)
                test_acc = elm.accuracy(X_test, y_test)
                results.append(
                    {
                        "activation": key,
                        "seed": seed,
                        "hidden_size": hidden_size,
                        "train_accuracy": train_acc,
                        "test_accuracy": test_acc,
                    }
                )
    df = pl.DataFrame(results)
    print(df)
    df.write_csv("elm_mnist.csv")
    comparison_df = (
        df.group_by("activation", "hidden_size")
        .agg(
            train_accuracy_mean=pl.mean("train_accuracy"),
            train_accuracy_std=pl.std("train_accuracy"),
            test_accuracy_mean=pl.mean("test_accuracy"),
            test_accuracy_std=pl.std("test_accuracy"),
        )
        .sort(["activation", "hidden_size"])
    )
    comparison_df.write_csv("elm_mnist_comparison.csv")
    print(comparison_df)
# %%
