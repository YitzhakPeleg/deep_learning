import struct
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np


class Split(StrEnum):
    TRAIN = "train"
    TEST = "test"


_MAGIC_IMAGES = 2051
_MAGIC_LABELS = 2049

_FILENAMES: MappingProxyType[Split, tuple[str, str]] = MappingProxyType({
    Split.TRAIN: ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    Split.TEST:  ("t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte"),
})


def _read_images(path: Path) -> np.ndarray:
    """Read an IDX image file into a uint8 array.

    Parameters
    ----------
    path : Path
        Path to an uncompressed IDX3 image file.

    Returns
    -------
    np.ndarray
        Shape ``(N, rows, cols)``, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file's magic number is not 2051.
    """
    with path.open("rb") as f:
        # ">IIII": big-endian (>), four unsigned 32-bit ints (I):
        #   magic number, number of images, number of rows, number of columns
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != _MAGIC_IMAGES:
            raise ValueError(f"Expected magic {_MAGIC_IMAGES}, got {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    """Read an IDX label file into a uint8 array.

    Parameters
    ----------
    path : Path
        Path to an uncompressed IDX1 label file.

    Returns
    -------
    np.ndarray
        Shape ``(N,)``, dtype ``uint8``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file's magic number is not 2049.
    """
    with path.open("rb") as f:
        # ">II": big-endian (>), two unsigned 32-bit ints (I):
        #   magic number, number of labels
        magic, n = struct.unpack(">II", f.read(8))
        if magic != _MAGIC_LABELS:
            raise ValueError(f"Expected magic {_MAGIC_LABELS}, got {magic} in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(
    data_dir: Path | str,
    split: Split | str = Split.TRAIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST images and labels from IDX binary files.

    Parameters
    ----------
    data_dir : Path | str
        Directory containing the uncompressed IDX binary files.
    split : Split | str
        Which split to load. Accepts ``Split`` members or plain strings
        ``"train"`` / ``"test"``.

    Returns
    -------
    images : np.ndarray
        Shape ``(N, 28, 28)``, dtype ``uint8``. Raw pixel values in [0, 255].
    labels : np.ndarray
        Shape ``(N,)``, dtype ``uint8``. Digit class labels in [0, 9].

    Raises
    ------
    ValueError
        If ``split`` is not a valid ``Split`` value.
    FileNotFoundError
        If the expected IDX files are not present in ``data_dir``.
    ValueError
        If a file's magic number does not match the IDX spec.
    """
    split = Split(split)
    img_name, lbl_name = _FILENAMES[split]
    root = Path(data_dir)
    return _read_images(root / img_name), _read_labels(root / lbl_name)
