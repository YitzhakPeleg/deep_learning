"""Restricted Boltzmann Machine (RBM) and Deep Belief Network (DBN) for MNIST.

Architecture
------------
RBM  : energy-based undirected graphical model with visible layer ``v``
       (784-dim binarised pixel) and hidden layer ``h``.  Weights learned by
       Contrastive Divergence-k (CD-k).
DBN  : greedy layer-wise stack of RBMs.  Each layer is pre-trained on the
       mean activations of the layer below, then a linear classifier is
       attached and fine-tuned with cross-entropy + Adam.

Reference: Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). "A fast learning
algorithm for deep belief nets". Neural Computation, 18(7), 1527–1554.
"""
# %%

import sys
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import tqdm
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


class RBM:
    """Energy-based undirected model trained with Contrastive Divergence-k.

    Both visible and hidden units are Bernoulli (binary).  Input images are
    binarised by normalising to ``[0, 1]`` and sampling; continuous-valued
    mean-field activations are used for stacking inside a DBN.

    Parameters
    ----------
    visible_size : int
        Number of visible units (784 for flattened MNIST).
    hidden_size : int
        Number of hidden units.
    k : int
        Number of Gibbs steps in CD-k.  ``k=1`` is standard and fast;
        larger ``k`` gives a better gradient estimate at higher cost.
    learning_rate : float
        Step size applied to ``(positive phase - negative phase)`` gradients.
    seed : int | None
        Seed for NumPy RNG used to initialise weights and draw samples.
        ``None`` means non-reproducible.

    Attributes
    ----------
    W : np.ndarray | None
        Weight matrix, shape ``(visible_size, hidden_size)``.
        ``None`` before :meth:`fit`.
    v_bias : np.ndarray | None
        Visible bias vector, shape ``(visible_size,)``.
        ``None`` before :meth:`fit`.
    h_bias : np.ndarray | None
        Hidden bias vector, shape ``(hidden_size,)``.
        ``None`` before :meth:`fit`.
    """

    def __init__(
        self,
        visible_size: int,
        hidden_size: int,
        k: int = 1,
        learning_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(loc=0, scale=0.01, size=(visible_size, hidden_size))
        self.v_bias = np.zeros(visible_size)
        self.h_bias = np.zeros(hidden_size)
        self.k = k
        self.learning_rate = learning_rate

    def fit(
        self,
        X: np.ndarray,
        *,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> Self:
        """Train the RBM with Contrastive Divergence.

        For each mini-batch: compute the positive phase, run ``k`` steps of
        alternating Gibbs sampling for the negative phase, then update
        ``W``, ``v_bias``, and ``h_bias`` by gradient ascent on the log
        likelihood approximation.

        Parameters
        ----------
        X : np.ndarray
            Training images, shape ``(N, 28, 28)`` or ``(N, 784)``.
            Values in ``[0, 255]``; normalised internally to ``[0, 1]``.
        epochs : int
            Number of full passes over the training set.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        Self
            ``self``, for chaining.
        """
        return self._fit_normalized(
            self._prepare_input(X), epochs=epochs, batch_size=batch_size
        )

    def _fit_normalized(
        self,
        X_prepared: np.ndarray,
        *,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> Self:
        """Train on pre-normalised input, skipping ``_prepare_input``.

        Called by :meth:`fit` (after normalising) and by :class:`DBN` for
        intermediate layers whose input is already in ``[0, 1]``.

        Parameters
        ----------
        X_prepared : np.ndarray
            Training data, shape ``(N, visible_size)``, dtype ``float64``,
            values already in ``[0, 1]``.
        epochs : int
            Number of full passes over the training set.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        Self
            ``self``, for chaining.
        """
        n_samples = X_prepared.shape[0]
        for epoch in tqdm.tqdm(range(epochs), desc="Training RBM"):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_prepared[indices]

            loss = 0.0
            batches = 0
            for start in range(0, n_samples, batch_size):
                batch = X_shuffled[start : start + batch_size]
                loss += self._fit_batch(batch)
                batches += 1
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss / batches:.4f}")

        return self

    def _fit_batch(self, v: np.ndarray) -> float:
        """Update weights and biases using CD-k on one mini-batch.

        Args:
            v (np.ndarray): Mini-batch of visible activations, shape (batch_size, visible_size).
        """
        dw, dv_bias, dh_bias, loss = self._cd_step(v)
        self.W += self.learning_rate * dw
        self.v_bias += self.learning_rate * dv_bias
        self.h_bias += self.learning_rate * dh_bias
        return loss

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode inputs to mean hidden activations ``p(h=1 | v)``.

        Used when stacking RBMs inside a :class:`DBN`: the output is passed
        as the visible layer to the next RBM.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.

        Returns
        -------
        np.ndarray
            Mean hidden activations, shape ``(N, hidden_size)``, dtype ``float64``.
        """
        X_prepared = self._prepare_input(X)
        return self._calc_hidden_prob(X_prepared)

    def _calc_hidden_prob(self, v: np.ndarray) -> np.ndarray:
        """Compute hidden probabilities and draw a Bernoulli sample.

        ``p(h_j=1 | v) = sigma(h_bias_j + v W_{:,j})``

        Parameters
        ----------
        v : np.ndarray
            Visible activations, shape ``(batch, visible_size)``.

        Returns
        -------
        np.ndarray
            Mean hidden activations, shape ``(batch, hidden_size)``.
        """
        return self._calc_prob(energy=self.h_bias + v @ self.W)

    def _calc_visible_prob(self, h: np.ndarray) -> np.ndarray:
        """Compute visible probabilities and draw a Bernoulli sample.

        ``p(v_i=1 | h) = sigma(v_bias_i + h W_{i,:}ᵀ)``

        Parameters
        ----------
        h : np.ndarray
            Hidden activations, shape ``(batch, hidden_size)``.

        Returns
        -------
        np.ndarray
            Mean visible activations, shape ``(batch, visible_size)``.
        """
        return self._calc_prob(energy=self.v_bias + h @ self.W.T)

    def _sample(self, prob: np.ndarray) -> np.ndarray:
        return prob > self.rng.random(size=prob.shape)

    def _calc_prob(self, energy: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-energy))

    def _cd_step(
        self, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute CD-k weight gradients for one mini-batch.

        Runs one positive phase (data → hidden) and ``k`` alternating
        Gibbs steps (hidden → visible → hidden) for the negative phase,
        then returns:

        ``dW     = (v₀ h₀ᵀ - v_k h_kᵀ) / batch``
        ``dv_bias = (v₀ - v_k) / batch``
        ``dh_bias = (h₀ - h_k) / batch``

        Parameters
        ----------
        v : np.ndarray
            Mini-batch visible activations, shape ``(batch, visible_size)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(dW, dv_bias, dh_bias)`` with shapes matching ``W``, ``v_bias``,
            ``h_bias`` respectively.
        """
        v0 = v
        h0_prob = self._calc_hidden_prob(v)

        h_k_sample = self._sample(h0_prob)
        v_k_sample = v0
        for _ in range(self.k):
            v_k_prob = self._calc_visible_prob(h_k_sample)
            v_k_sample = self._sample(v_k_prob)
            h_k_prob = self._calc_hidden_prob(v_k_sample)
            h_k_sample = self._sample(h_k_prob)

        # v dim (batch, visible_size), h dim (batch, hidden_size)
        dW = (v.T @ h0_prob - v_k_sample.T @ h_k_prob) / v.shape[0]
        # dw dim (visible_size, hidden_size),
        dv_bias: np.ndarray = np.mean(
            v - v_k_sample, axis=0
        )  # dv_bias dim (visible_size,)
        dh_bias = np.mean(h0_prob - h_k_prob, axis=0)  # dh_bias dim (hidden_size,)
        loss = np.mean(np.abs(np.concat([dW.reshape(-1), dv_bias, dh_bias])))
        return dW, dv_bias, dh_bias, loss

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Flatten and normalise input images.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, visible_size)``, dtype ``float64``, values in ``[0, 1]``.
        """
        return X.reshape(X.shape[0], -1) / 255.0  # flatten


class DBN:
    """Deep Belief Network: greedy layer-wise RBM stack + linear classifier.

    Pre-training: each :class:`RBM` layer is trained on the ``transform()``
    output of the previous layer (raw pixels for layer 0).  Fine-tuning:
    a single ``nn.Linear`` classifier is trained on the encoded features
    with cross-entropy loss and Adam.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of all layers including the input.  For example
        ``[784, 500, 500, 2000]`` creates two stacked RBMs
        (784→500, 500→500) and a top RBM (500→2000).
    n_classes : int
        Number of output classes (10 for MNIST).
    k : int
        CD-k steps passed to every :class:`RBM`.
    rbm_lr : float
        Learning rate for RBM pre-training.
    clf_lr : float
        Learning rate for the fine-tuning classifier (Adam).
    seed : int | None
        Base seed; each RBM layer receives ``seed + layer_index``.

    Attributes
    ----------
    rbms : list[RBM]
        One RBM per consecutive pair in ``layer_sizes``.
        Empty before :meth:`fit`.
    classifier : nn.Linear | None
        Linear classifier mapping top hidden size → ``n_classes``.
        ``None`` before :meth:`fit`.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        n_classes: int = 10,
        k: int = 1,
        rbm_lr: float = 0.01,
        clf_lr: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        self.rbms = [
            RBM(
                visible_size=layer_sizes[i],
                hidden_size=layer_sizes[i + 1],
                k=k,
                learning_rate=rbm_lr,
                seed=(seed + i) if seed is not None else None,
            )
            for i in range(len(layer_sizes) - 1)
        ]
        self.classifier: nn.Linear | None = None
        self._top_size = layer_sizes[-1]
        self._n_classes = n_classes
        self.clf_lr = clf_lr

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        rbm_epochs: int = 10,
        clf_epochs: int = 20,
        batch_size: int = 64,
    ) -> Self:
        """Greedily pre-train RBMs then fine-tune the linear classifier.

        Stage 1 — pre-training: for each RBM layer, transform the training
        data through all previously fitted layers and call
        ``rbm.fit(transformed, epochs=rbm_epochs, batch_size=batch_size)``.

        Stage 2 — fine-tuning: encode all training data with :meth:`_encode`,
        then optimise a cross-entropy loss on the frozen features with Adam
        for ``clf_epochs`` passes.

        Parameters
        ----------
        X : np.ndarray
            Training images, shape ``(N, 28, 28)`` or ``(N, 784)``.
        y : np.ndarray
            Integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.
        rbm_epochs : int
            Epochs per RBM layer during pre-training.
        clf_epochs : int
            Epochs for classifier fine-tuning.
        batch_size : int
            Mini-batch size used in both stages.

        Returns
        -------
        Self
            ``self``, for chaining.
        """
        X_curr = self._prepare_input(X)
        for rbm in self.rbms:
            rbm._fit_normalized(X_curr, epochs=rbm_epochs, batch_size=batch_size)
            X_curr = rbm._calc_hidden_prob(X_curr)

        H = self._encode(X)
        H_t = torch.from_numpy(H).float()
        y_t = torch.from_numpy(y.astype(np.int64))

        self.classifier = nn.Linear(self._top_size, self._n_classes)
        opt = torch.optim.Adam(self.classifier.parameters(), lr=self.clf_lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in tqdm.tqdm(range(clf_epochs), desc="Training classifier"):
            perm = torch.randperm(len(H_t))
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(H_t), batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(self.classifier(H_t[idx]), y_t[idx])
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                batches += 1
            print(f"Epoch {epoch + 1}/{clf_epochs}, Loss: {epoch_loss / batches:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted class label for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(N,)``, dtype ``int64``.
        """
        if self.classifier is None:
            raise RuntimeError("DBN is not fitted yet. Call fit() before predicting.")
        H = self._encode(X)
        with torch.no_grad():
            logits = self.classifier(torch.from_numpy(H).float())
        return logits.argmax(dim=1).numpy().astype(np.int64)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute fraction of correctly classified samples.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.
        y : np.ndarray
            True labels, shape ``(N,)``.

        Returns
        -------
        float
            Accuracy in ``[0.0, 1.0]``.
        """
        return float(np.mean(self.predict(X) == y))

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """Pass input through all RBM layers using ``transform()``.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.

        Returns
        -------
        np.ndarray
            Top-layer hidden activations, shape ``(N, layer_sizes[-1])``,
            dtype ``float64``.
        """
        H = self._prepare_input(X)
        for rbm in self.rbms:
            H = rbm._calc_hidden_prob(H)
        return H

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Flatten and normalise input images.

        Parameters
        ----------
        X : np.ndarray
            Input images, shape ``(N, 28, 28)`` or ``(N, 784)``.

        Returns
        -------
        np.ndarray
            Shape ``(N, layer_sizes[0])``, dtype ``float64``, values in ``[0, 1]``.
        """
        return X.reshape(X.shape[0], -1) / 255.0


# %%
if __name__ == "__main__":
    from beyond_backprop.constants import DATA_PATH
    from beyond_backprop.mnist import load_mnist

    X_train, y_train = load_mnist(data_dir=DATA_PATH, split="train")
    X_test, y_test = load_mnist(data_dir=DATA_PATH, split="test")
    logger.info(f"train: {X_train.shape}, test: {X_test.shape}")

    dbn = DBN(
        layer_sizes=[784, 500, 500, 2000],
        n_classes=10,
        k=1,
        rbm_lr=0.01,
        clf_lr=1e-3,
        seed=0,
    ).fit(X_train, y_train, rbm_epochs=10, clf_epochs=20, batch_size=64)

    logger.info(f"train accuracy: {dbn.accuracy(X_train, y_train):.4f}")
    logger.info(f"test  accuracy: {dbn.accuracy(X_test, y_test):.4f}")
# %%
