"""Extreme Learning Machine (ELM) for MNIST classification.

Architecture
------------
Input → random frozen hidden layer → analytic output weights (ridge regression).

Reference: Huang et al. (2006), "Extreme learning machine: Theory and
applications", Neurocomputing.
"""

from enum import Enum
from typing import Self

import numpy as np
import torch


class Activation(Enum):
    SIGMOID = torch.nn.Sigmoid()
    TANH = torch.nn.Tanh()
    RELU = torch.nn.ReLU()


def _activate(Z: np.ndarray, activation: Activation) -> np.ndarray:
    """Apply elementwise activation function.

    Parameters
    ----------
    Z : np.ndarray
        Pre-activation matrix, any shape.
    activation : Activation
        Name of the activation function (``"sigmoid"`` or ``"tanh"``).

    Returns
    -------
    np.ndarray
        Same shape as ``Z``, dtype ``float64``.

    Raises
    ------
    ValueError
        If ``activation`` is not a supported value.
    """
    raise NotImplementedError


class ELM:
    """Single-hidden-layer feedforward network trained by ridge regression.

    Only the output weights ``beta`` are learned; the hidden-layer weights
    ``W1`` and biases ``b`` are sampled once at construction and never updated.

    Parameters
    ----------
    hidden_size : int
        Number of hidden neurons (``L`` in the doc).  More neurons → better
        representation up to the point where regularization is needed.
    activation : Activation
        Hidden-layer activation.  Use ``"sigmoid"`` or ``"tanh"``; avoid
        ``"relu"`` (permanently dead neurons with random weights, see doc).
    lambda_reg : float
        Ridge regularization coefficient ``λ``.  Added to the diagonal of
        ``HᵀH`` before solving to improve conditioning and reduce overfitting.
    random_seed : int | None
        Seed for the NumPy RNG used to draw ``W1`` and ``b``.  ``None`` means
        non-reproducible.

    Attributes
    ----------
    W1 : np.ndarray
        Hidden-layer weight matrix, shape ``(n_features, hidden_size)``.
        Set during :meth:`fit`; ``None`` before first call.
    b : np.ndarray
        Hidden-layer bias vector, shape ``(hidden_size,)``.
        Set during :meth:`fit``; ``None`` before first call.
    beta : np.ndarray
        Output weight matrix, shape ``(hidden_size, n_classes)``.
        Set during :meth:`fit`; ``None`` before first call.
    n_classes : int | None
        Number of output classes inferred from ``y`` at fit time.
    """

    W1: np.ndarray | None
    b: np.ndarray | None
    beta: np.ndarray | None
    n_classes: int | None

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: Activation,
        regularization_factor: float,
        seed: int | None = None,
    ) -> None:
        rng = np.random.RandomState(seed)
        self._hidden_layer = torch.nn.Linear(input_size, hidden_size)
        self._hidden_layer.weight.data = torch.nn.Parameter(
            data=torch.from_numpy(rng.normal(size=(hidden_size, input_size)))
        )
        self._hidden_layer.bias.data = torch.nn.Parameter(
            data=torch.from_numpy(rng.normal(size=(hidden_size,)))
        )
        self._activation = activation.value
        self._regularization_factor = regularization_factor
        self.n_classes = output_size
        self.output_layer = torch.nn.Linear(hidden_size, self.n_classes, bias=False)

        self.model = torch.nn.Sequential(
            self._hidden_layer, self._activation, self.output_layer
        )

    def __repr__(self) -> str:
        return self.model.__repr__()

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Draw random weights and solve for output weights analytically.

        Draws ``W1`` and ``b`` from a standard normal distribution, computes
        ``H = activation(X W1 + b)``, then solves::

            β = (HᵀH + λI)⁻¹ Hᵀ Y

        where ``Y`` is the one-hot encoding of ``y``.

        Parameters
        ----------
        X : np.ndarray
            Training images, shape ``(N, 784)`` (flattened), dtype ``float64``.
        y : np.ndarray
            Integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.

        Returns
        -------
        ELM
            ``self``, for chaining.
        """
        H = self._activation(self._hidden_layer(torch.from_numpy(X))).detach().numpy()
        # compute Moore-Penrose pseudoinverse of H with ridge regularization
        U, S, Vt = np.linalg.svd(H, full_matrices=False)
        # S_inv = 1 / S with ridge regularization to improve conditioning and reduce overfitting
        S_inv = S / (S**2 + self._regularization_factor)
        # Compute pseudoinverse of H: H⁺ = V S⁻¹ Uᵀ
        H_pinv = Vt.T @ np.diag(S_inv) @ U.T
        #
        Y = self._one_hot(y)
        # Solve for output weights: β = H⁺ Y
        self.output_layer.weight.data = torch.nn.Parameter(
            data=torch.from_numpy(H_pinv @ Y)
        )
        # return self for chaining (e.g. ELM(...).fit(X_train, y_train).accuracy(X_test, y_test))
        return self

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute raw (unnormalized) class scores for each sample.

        Returns ``H β`` without a final softmax so that the caller can apply
        argmax directly for hard predictions or softmax for probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 784)``, dtype ``float64``.

        Returns
        -------
        np.ndarray
            Score matrix, shape ``(N, n_classes)``, dtype ``float64``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        return self.model(torch.from_numpy(X)).detach().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 784)``, dtype ``float64``.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(N,)``, dtype ``int64``.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute fraction of correctly classified samples.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 784)``, dtype ``float64``.
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
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from beyond_backprop.constants import DATA_PATH
    from beyond_backprop.mnist import load_mnist

    X_train, y_train = load_mnist(data_dir=DATA_PATH, split="train")

    elm = ELM(
        input_size=784,
        hidden_size=1000,
        output_size=10,
        activation=Activation.SIGMOID,
        regularization_factor=1e-3,
        seed=42,
    )
    print(elm)
    elm.fit(X_train, y_train)
