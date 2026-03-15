"""Sparse and dense block matrix utilities."""

from __future__ import annotations

import numpy as np


def build_block_matrix(blocks: list[list[np.ndarray]]) -> np.ndarray:
    """Assemble a dense block matrix from a list of row-lists of arrays.

    Args:
        blocks: List of rows; each row is a list of 2D numpy arrays with
                the same number of rows.  E.g. ``[[A, B], [C, D]]``.

    Returns:
        Dense numpy array containing the assembled block matrix.

    Example:
        >>> A = np.eye(3); B = np.ones((3, 2))
        >>> build_block_matrix([[A, B]])
        array([[1., 0., 0., 1., 1.],
               [0., 1., 0., 1., 1.],
               [0., 0., 1., 1., 1.]])
    """
    return np.block(blocks)
