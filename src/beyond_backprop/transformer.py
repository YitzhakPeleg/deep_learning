"""Transformer classifiers for MNIST: row-tokenized and ViT (patch) variants.

Architecture
------------
Both variants share the same encoder stack (multi-head self-attention + FFN
blocks with pre-norm residuals) and classification head (CLS token → linear).
They differ only in how the 28×28 image is split into tokens.

- RowTransformer  : 28 row-tokens, each of dim 28  (T=28, d_token=28)
- PatchTransformer: non-overlapping square patches  (default patch_size=4
                    → T=49 tokens of dim 16)

Reference: Vaswani et al. (2017), "Attention Is All You Need".
           Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words" (ViT).
"""
# %%

import sys
from typing import Self

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

logger.remove()
logger.add(sys.stderr, level="INFO")


# ── building blocks ────────────────────────────────────────────────────────────


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with a projection output.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.  Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads.  Each head operates on dimension
        ``d_model // n_heads``.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : Tensor
            Input sequence, shape ``(B, T, d_model)``.

        Returns
        -------
        out : Tensor
            Attended output, shape ``(B, T, d_model)``.
        attn_weights : Tensor
            Averaged attention weights across heads, shape ``(B, T, T)``.
            Useful for visualization.
        """
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """One Transformer encoder block: pre-norm self-attention + pre-norm FFN.

    Equations (pre-norm variant)::

        Z' = Z  + MHA(LayerNorm(Z))
        Z''= Z' + FFN(LayerNorm(Z'))

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_ff : int
        Inner dimension of the feed-forward sublayer (typically 4 * d_model).
    dropout : float
        Dropout applied inside attention and FFN.
    """

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run one encoder block.

        Parameters
        ----------
        x : Tensor
            Input, shape ``(B, T, d_model)``.

        Returns
        -------
        out : Tensor
            Output, shape ``(B, T, d_model)``.
        attn_weights : Tensor
            Attention weights from the MHA sublayer, shape ``(B, T, T)``.
        """
        raise NotImplementedError


# ── shared encoder base ────────────────────────────────────────────────────────


class _TransformerBase(nn.Module):
    """Shared encoder stack + classification head used by both variants.

    Subclasses must implement ``_tokenize`` to convert a raw image batch into
    a token sequence before the encoder is applied.

    Parameters
    ----------
    d_token : int
        Raw token dimension (before projection to ``d_model``).
    seq_len : int
        Number of tokens per image (before prepending CLS).
    d_model : int
        Model (embedding) dimension.
    n_heads : int
        Number of attention heads.
    n_blocks : int
        Number of stacked TransformerBlocks.
    d_ff : int
        FFN inner dimension per block.
    n_classes : int
        Output classes (10 for MNIST).
    dropout : float
        Dropout probability.
    seed : int | None
        Seed for ``torch.manual_seed`` at construction for reproducibility.
    """

    def __init__(
        self,
        d_token: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_blocks: int,
        d_ff: int,
        n_classes: int,
        dropout: float,
        seed: int | None,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def _tokenize(self, images: Tensor) -> Tensor:
        """Convert a batch of images to a token sequence.

        Parameters
        ----------
        images : Tensor
            Raw images, shape ``(B, 28, 28)``.

        Returns
        -------
        Tensor
            Token sequence, shape ``(B, seq_len, d_token)``.
        """
        raise NotImplementedError

    def forward(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the full encoder and return logits + per-block attention weights.

        Parameters
        ----------
        images : Tensor
            Raw images, shape ``(B, 28, 28)``.

        Returns
        -------
        logits : Tensor
            Shape ``(B, n_classes)``.
        attn_weights : list[Tensor]
            One ``(B, T+1, T+1)`` tensor per block (T+1 because CLS is prepended).
        """
        raise NotImplementedError

    # ── sklearn-style training interface ──────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        batch_size: int = 256,
        epochs: int = 20,
        lr: float = 3e-4,
        warmup_fraction: float = 0.1,
        device: str | None = None,
    ) -> Self:
        """Train on MNIST with Adam + linear LR warmup and cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Images, shape ``(N, 28, 28)`` or ``(N, 784)``, values in ``[0, 1]``.
        y : np.ndarray
            Integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.
        batch_size : int
            Mini-batch size.
        epochs : int
            Number of full passes over the training set.
        lr : float
            Peak learning rate for Adam.
        warmup_fraction : float
            Fraction of total steps over which the LR linearly ramps from 0 to
            ``lr``.  After warmup, LR stays constant (no decay).
        device : str | None
            PyTorch device string.  ``None`` selects CUDA if available, else CPU.

        Returns
        -------
        Self
            ``self``, for chaining.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray, *, device: str | None = None) -> np.ndarray:
        """Classify images.

        Parameters
        ----------
        X : np.ndarray
            Images, shape ``(N, 28, 28)`` or ``(N, 784)``, values in ``[0, 1]``.
        device : str | None
            PyTorch device string.

        Returns
        -------
        np.ndarray
            Integer class predictions, shape ``(N,)``, dtype ``int64``.
        """
        raise NotImplementedError

    def accuracy(
        self, X: np.ndarray, y: np.ndarray, *, device: str | None = None
    ) -> float:
        """Fraction of correctly classified images.

        Parameters
        ----------
        X : np.ndarray
            Images, shape ``(N, 28, 28)`` or ``(N, 784)``.
        y : np.ndarray
            True integer labels, shape ``(N,)``.
        device : str | None
            PyTorch device string.

        Returns
        -------
        float
            Accuracy in ``[0.0, 1.0]``.
        """
        raise NotImplementedError

    def attention_maps(
        self, X: np.ndarray, *, device: str | None = None
    ) -> list[np.ndarray]:
        """Return per-block attention weight matrices for a batch of images.

        Parameters
        ----------
        X : np.ndarray
            Images, shape ``(N, 28, 28)`` or ``(N, 784)``.
        device : str | None
            PyTorch device string.

        Returns
        -------
        list[np.ndarray]
            One array per block, each of shape ``(N, T+1, T+1)`` where T+1
            is the sequence length including the CLS token.  Values are
            attention probabilities (post-softmax) averaged over heads.
        """
        raise NotImplementedError


# ── concrete variants ──────────────────────────────────────────────────────────


class RowTransformer(_TransformerBase):
    """Transformer that treats each pixel row as a token (T=28, d_token=28).

    The 28×28 image is viewed as a sequence of 28 row-vectors, each of length
    28.  A CLS token is prepended, giving a sequence of length 29 entering
    the encoder.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.  Default 64.
    n_heads : int
        Number of attention heads.  Must divide ``d_model``.  Default 4.
    n_blocks : int
        Number of stacked encoder blocks.  Default 2.
    d_ff : int
        FFN inner dimension.  Default 256 (= 4 × d_model).
    n_classes : int
        Output classes.  Default 10.
    dropout : float
        Dropout probability.  Default 0.0.
    seed : int | None
        Reproducibility seed.  Default ``None``.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        d_ff: int = 256,
        n_classes: int = 10,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            d_token=28,
            seq_len=28,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            n_classes=n_classes,
            dropout=dropout,
            seed=seed,
        )

    def _tokenize(self, images: Tensor) -> Tensor:
        """Split each image into 28 row tokens of dim 28.

        Parameters
        ----------
        images : Tensor
            Shape ``(B, 28, 28)``.

        Returns
        -------
        Tensor
            Shape ``(B, 28, 28)`` — the image is already in token form.
        """
        raise NotImplementedError


class PatchTransformer(_TransformerBase):
    """ViT-style Transformer that splits each image into non-overlapping patches.

    With the default ``patch_size=4``, the 28×28 image yields 7×7 = 49 patches
    each of dimension 4×4 = 16, giving T=49 tokens of dim 16.

    Parameters
    ----------
    patch_size : int
        Side length of each square patch.  Must evenly divide 28.  Default 4.
    d_model : int
        Model (embedding) dimension.  Default 64.
    n_heads : int
        Number of attention heads.  Must divide ``d_model``.  Default 4.
    n_blocks : int
        Number of stacked encoder blocks.  Default 2.
    d_ff : int
        FFN inner dimension.  Default 256.
    n_classes : int
        Output classes.  Default 10.
    dropout : float
        Dropout probability.  Default 0.0.
    seed : int | None
        Reproducibility seed.  Default ``None``.
    """

    def __init__(
        self,
        patch_size: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        d_ff: int = 256,
        n_classes: int = 10,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if 28 % patch_size != 0:
            raise ValueError(f"patch_size={patch_size} must evenly divide 28")
        n_patches_per_side = 28 // patch_size
        seq_len = n_patches_per_side * n_patches_per_side
        d_token = patch_size * patch_size
        self.patch_size = patch_size
        super().__init__(
            d_token=d_token,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            n_classes=n_classes,
            dropout=dropout,
            seed=seed,
        )

    def _tokenize(self, images: Tensor) -> Tensor:
        """Extract non-overlapping patches and flatten each to a vector.

        Parameters
        ----------
        images : Tensor
            Shape ``(B, 28, 28)``.

        Returns
        -------
        Tensor
            Shape ``(B, seq_len, patch_size**2)`` — raster-order patches.
        """
        raise NotImplementedError
