"""Transformer classifiers for MNIST: row-tokenized and ViT (patch) variants.

Architecture
------------
Both variants share the same encoder stack (multi-head self-attention + FFN
blocks with pre-norm residuals) and classification head (CLS token → linear).
They differ only in how the 28 * 28 image is split into tokens.

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
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

logger.remove()
logger.add(sys.stderr, level="INFO")


# ── building blocks ────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """Scaled dot-product self-attention with separate query/key/value projections.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.norm_factor = d_model**0.5  # for scaled dot-product attention
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Compute self-attention.

        Parameters
        ----------
        x : Tensor
            Input sequence, shape ``(B, T, d_model)``.

        Returns
        -------
        out : Tensor
            Attended output, shape ``(B, T, d_model)``.
        attn_weights : Tensor
            Attention weights, shape ``(B, T, T)``.  Useful for visualization.
        """
        attention = self.attention_matrix(x) @ self.V(x)
        return self.dropout_layer(attention)

    def attention_matrix(self, x: Tensor) -> Tensor:
        """Compute the attention weight matrix (post-softmax) for a batch of input

        Args:
            x (Tensor): Input sequence, shape ``(B, T, d_model)``.

        Returns:
            Tensor: Attention weights, shape ``(B, T, T)``.  Useful for visualization.
        """
        z_q: Tensor = self.Q(x)
        z_k: Tensor = self.K(x)
        attn_weights = torch.softmax(
            z_q @ z_k.transpose(1, 2) / self.norm_factor, dim=2
        )
        return attn_weights


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
        if d_model % n_heads != 0:
            raise ValueError(f"{d_model=} must be divisible by {n_heads=}")
        self.head_dim = d_model // n_heads
        self.heads = nn.ModuleList(
            [SelfAttention(self.head_dim, dropout) for _ in range(n_heads)]
        )
        self.dropout = nn.Dropout(dropout)
        self.final_projections = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
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
        multihead_attention = torch.cat(
            [
                attention_head(head_input)
                for attention_head, head_input in zip(
                    self.heads,
                    self._distribute_to_heads(x),
                )
            ],
            dim=-1,
        )
        projected_attention = self.final_projections(multihead_attention)
        return self.dropout(projected_attention)

    def _distribute_to_heads(self, x: Tensor) -> list[Tensor]:
        """Split the input into separate heads for multi-head attention.

        Parameters
        ----------
        x : Tensor
            Input sequence, shape ``(B, seq_len, d_model)``.
        Returns
        -------
        list[Tensor]
            List of length ``n_heads``, each tensor of shape ``(B, seq_len, n_heads, self.head_dim)``.
        """
        batch_size, seq_len, _ = x.shape
        # reshape to (B, seq_len, n_heads, head_dim) and split along the last dimension to get a list of head inputs
        # heads must be before head_dim in the reshape so that the head_dim is contiguous in memory for each head, which allows us to split it correctly into separate heads
        # so [1,2,3,4,5,6] with n_heads=2 would reshape [[1,2,3],[4,5,6]]
        # where the first head gets [1,2,3]
        # and the second head gets [4,5,6]
        head_inputs = x.view(batch_size, seq_len, len(self.heads), self.head_dim)
        return [head_inputs[:, :, i, :] for i in range(len(self.heads))]


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
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.multihead_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
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

        # claculate the output of the multi-head attention sublayer and add the residual connection
        Z_prime = x + self.multihead_attention(self.layer_norm_1(x))
        # calculate the output of the feed-forward sublayer and add the residual connection
        Z_double_prime = Z_prime + self.feed_forward(self.layer_norm_2(Z_prime))
        # return the output of the block after applying dropout, and the attention weights for visualization
        return self.dropout(Z_double_prime)


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
        d_model: int,
        n_heads: int,
        n_blocks: int,
        d_ff: int,
        n_classes: int,
        dropout: float,
        seed: int | None,
    ) -> None:
        if seed is not None:
            logger.info(f"Setting random seed to {seed} for reproducibility")
            torch.manual_seed(seed)
        super().__init__()
        # transformer blocks for encoding the token sequence
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)]
        )
        # final linear layer for classification from CLS token to output classes
        self.final_linear_layer = nn.Linear(d_model, n_classes)
        self.d_model = d_model
        # cls_token is a learnable parameter of shape (1, 1, d_model)
        # (it's a single toekn with batch size 1 - we later expand it to match the batch size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _tokenize(self, x: Tensor) -> Tensor:
        """Tokenize the input sequence.

        Parameters
        ----------
        x : Tensor
            Input tokens, shape ``(B, seq_len, d_token)``.

        Returns
        -------
        Tensor
            Token sequence, shape ``(B, seq_len, d_model)``.
        """
        raise NotImplementedError

    def _positional_encoding(self, tokens: Tensor) -> Tensor:
        """Add positional encoding to the token sequence.

        Parameters
        ----------
        tokens : Tensor
            Token sequence, shape ``(B, seq_len, d_model)``.

        Returns
        -------
        Tensor
            Positionally encoded tokens, shape ``(B, seq_len, d_model)``.
        """
        seq_len = tokens.shape[1]
        # create a positional encoding tensor of shape (seq_len, d_model)
        position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        # 1e4 ^ (2i/d_model) == exp(log(1e4) * 2i/d_model) == exp(2i * log(1e4) / d_model)
        c = np.log(1e4) / tokens.shape[2]  # log(1e4) / d_model
        # one_over_denominator is the term that multiplies the position in the sin/cos formulas
        one_over_denominator = torch.exp(torch.arange(0, self.d_model, 2) * -c)
        # calculate the positional encoding using sin for even indices and cos for odd indices
        pe = torch.zeros(seq_len, self.d_model)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(
            position * one_over_denominator
        )  # apply sin to even indices
        pe[:, 1::2] = torch.cos(
            position * one_over_denominator
        )  # apply cos to odd indices
        pe = pe.unsqueeze(0)  # (1, seq_len, d_token)
        return pe.to(tokens.device)  # add positional encoding to tokens

    def forward(self, images: Tensor) -> Tensor:
        """Run the full encoder and return logits.

        Parameters
        ----------
        images : Tensor
            Raw images, shape ``(B, 28, 28)``.

        Returns
        -------
        logits : Tensor
            Shape ``(B, n_classes)``.
        """
        as_tokens = self._tokenize(images)
        # prepend CLS token
        cls_token = self.cls_token.expand(as_tokens.shape[0], -1, -1)
        # prepend - CLS is another token in the sequence (dim=1)
        tokens_with_cls = torch.cat([cls_token, as_tokens], dim=1)
        # shape of tokens_with_cls is now (B, seq_len+1, d_token) - the +1 is for the CLS token
        # add positional encoding to the tokens (the CLS token gets position 0, the first image token gets position 1, etc.)
        tokens_with_cls += self._positional_encoding(tokens_with_cls)
        # run tokens through the encoder blocks
        for block in self.transformer_blocks:
            tokens_with_cls = block(tokens_with_cls)
        # final linear layer on the CLS token for classification
        logits = self.final_linear_layer(tokens_with_cls[:, 0, :])
        return logits

    # ── sklearn-style training interface ──────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        batch_size: int = 256,
        epochs: int = 20,
        lr: float = 3e-4,
        warmup_fraction: float = 0.1,
        device: str | None = None,
    ) -> Self:
        """Train on MNIST with Adam + linear LR warmup and cross-entropy loss.

        Parameters
        ----------
        X_train : np.ndarray
            Training images, shape ``(N, 28, 28)`` or ``(N, 784)``, values in ``[0, 1]``.
        y_train : np.ndarray
            Training integer class labels, shape ``(N,)``, values in ``[0, n_classes)``.
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
        self.train()
        # set an optimizer (Adam) and a loss function (cross-entropy)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-16,  # start with a very small learning rate to avoid instability at the beginning of training
            end_factor=1.0,  # end with the specified learning rate
            total_iters=int(warmup_fraction * epochs),
            last_epoch=-1,
        )
        loss_function = nn.CrossEntropyLoss()
        NUM_EPOCHS = epochs
        loss_values = np.empty(NUM_EPOCHS)
        NUM_BATCHES = len(X_train) // batch_size
        logger.info(
            f"Starting training for {NUM_EPOCHS} epochs {NUM_BATCHES} batches per epoch"
        )
        run_evaluation = X_eval is not None and y_eval is not None
        if run_evaluation:
            logger.info("Evaluation will be run at the end of each epoch")
            eval_loss_values = np.empty_like(loss_values)
        for iteration in range(NUM_EPOCHS):
            self.train()
            scheduler.step()  # update the learning rate according to the scheduler
            logger.info(
                f"Epoch {iteration} starting with learning rate: {scheduler.get_last_lr()[0]:.4f}"
            )
            epoch_loss = 0.0
            batches_indices = np.random.permutation(
                # make sure to only take a multiple of batch_size to avoid having a smaller batch at the end
                NUM_BATCHES * batch_size
            ).reshape(-1, batch_size)  # shuffle the data at the start of each epoch
            tq_batch = tqdm(batches_indices, desc="Training (Batch)", unit="step")
            for batch_indeices in tq_batch:
                X_batch = X_train[batch_indeices]
                y_batch = y_train[batch_indeices]
                # # move data to the specified device
                # if device:
                #     X_batch = torch.from_numpy(X_batch).float().to(device)
                #     y_batch = torch.from_numpy(y_batch).long().to(device)
                # zero the gradients before starting training
                optimizer.zero_grad()
                # run forward pass
                logits = self.forward(torch.from_numpy(X_batch).float())
                # calculate the loss
                # why long? because CrossEntropyLoss expects the target labels to be of type long (int64) since they are class indices
                loss = loss_function(logits, torch.from_numpy(y_batch).long())
                # run backward pass to calculate gradients
                loss.backward()
                # update the parameters using the optimizer
                optimizer.step()
                # save loss to loguru for visualization
                epoch_loss += loss.item()
            mean_epoch_loss = epoch_loss / NUM_BATCHES
            logger.info(f"Training | epoch={iteration} | loss={mean_epoch_loss:.4f}")
            loss_values[iteration] = mean_epoch_loss
            if run_evaluation:
                self.eval()
                with torch.no_grad():
                    eval_loss_values[iteration] = loss_function(
                        self.forward(torch.from_numpy(X_eval).float()),
                        torch.from_numpy(y_eval).long(),
                    ).item()
                logger.info(
                    f"Evaluation | epoch={iteration} | loss={eval_loss_values[iteration]:.4f}"
                )
        self.eval()  # set the model to evaluation mode after training is done
        return self

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
        # run forward pass and take max over the classes (output of forward is (N, n_classes))
        with torch.no_grad():
            return self.forward(torch.from_numpy(X).float()).argmax(dim=1).numpy()

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
        pred = self.predict(X, device=device)
        return float(np.mean(pred == y))

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

    The 28*28 image is viewed as a sequence of 28 row-vectors, each of length
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
        FFN inner dimension.  Default 256 (= 4 * d_model).
    n_classes : int
        Output classes.  Default 10.
    dropout : float
        Dropout probability.  Default 0.0.
    seed : int | None
        Reproducibility seed.  Default ``None``.
    """

    def __init__(
        self,
        d_token: int = 28,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        d_ff: int = 256,
        n_classes: int = 10,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            n_classes=n_classes,
            dropout=dropout,
            seed=seed,
        )
        self.tokenizer = nn.Linear(d_token, d_model)

    def _tokenize(self, images: Tensor) -> Tensor:
        """Split each image into 28 row tokens of dim 28.

        Parameters
        ----------
        images : Tensor
            Shape ``(B, 28, 28)``.

        Returns
        -------
        Tensor
            Shape ``(B, seq_len, d_model)`` — the image is already in token form.
        """
        # appliy tokenizer layer to each token (dim=-1) to project from d_token to d_model
        # Linear layer applies to the last dimensions, so we can just pass the
        # image tensor directly since it's already in the shape (B, 28, 28)
        # where the last dimension is d_token=28
        return self.tokenizer(images)


class PatchTransformer(_TransformerBase):
    """ViT-style Transformer that splits each image into non-overlapping patches.

    With the default ``patch_size=4``, the 28*28 image yields 7*7 = 49 patches
    each of dimension 4*4 = 16, giving T=49 tokens of dim 16.

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
        patch_size: int | tuple[int, int] = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 2,
        d_ff: int = 256,
        n_classes: int = 10,
        dropout: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            n_blocks=n_blocks,
            d_ff=d_ff,
            n_classes=n_classes,
            dropout=dropout,
            seed=seed,
        )
        self.tokenizer = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def _tokenize(self, x: Tensor) -> Tensor:
        """Extract non-overlapping patches and flatten each to a vector.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, 28, 28)``.

        Returns
        -------
        Tensor
            Shape ``(B, seq_len, d_model)`` — raster-order patches.
            seq_len is the number of patches (e.g. 49 for patch_size=4)
            and d_model is the output dimension of the tokenizer layer
            (not the raw token dimension, since the Conv2d directly outputs the model dimension).
        """
        # add a new dim (1) for the channels so the input is (B, 1, 28, 28)
        # and apply the Conv2d tokenizer to extract patches and project to d_model dimension
        y: Tensor = self.tokenizer(x.unsqueeze(1))
        # now the last 2 dims holds all the patches in a grid format,
        # so we need to rearrange them to be a sequence of patches (tokens)
        z = y.reshape(shape=(y.shape[0], y.shape[1], -1))
        # shape of z is now (B, d_model, seq_len) where seq_len is the number of patches,
        # so we need to transpose to get (B, seq_len, d_model)
        return z.transpose(1, 2)


# %%
if __name__ == "__main__":
    from beyond_backprop.constants import DATA_PATH
    from beyond_backprop.mnist import load_mnist

    train_images, train_labels = load_mnist(
        DATA_PATH, split="train"
    )  # shape (N, 28, 28)
    train_images = train_images / 255.0  # normalize to [0, 1]
    N_eval = len(train_images) // 5  # use 20% of the training data for evaluation
    logger.info(f"Number of evaluation samples: {N_eval}")
    eval_images, eval_labels = train_images[:N_eval], train_labels[:N_eval]
    train_images, train_labels = train_images[N_eval:], train_labels[N_eval:]
    test_images, test_labels = load_mnist(DATA_PATH, split="test")  # shape (N, 28, 28)
    test_images = test_images / 255.0  # normalize to [0, 1]

    row_transformer = RowTransformer(seed=42).fit(
        X_train=train_images,
        y_train=train_labels,
        X_eval=eval_images,
        y_eval=eval_labels,
        epochs=20,
        lr=1e-3,
        warmup_fraction=0.1,
    )

    patch_transformer = PatchTransformer(seed=42).fit(
        X_train=train_images,
        y_train=train_labels,
        X_eval=eval_images,
        y_eval=eval_labels,
        epochs=20,
        lr=1e-3,
        warmup_fraction=0.1,
    )

    logger.info(
        f"Test accuracy {row_transformer.__class__.__name__}: {row_transformer.accuracy(test_images, test_labels):.4f}"
    )
    logger.info(
        f"Test accuracy {patch_transformer.__class__.__name__}: {patch_transformer.accuracy(test_images, test_labels):.4f}"
    )
