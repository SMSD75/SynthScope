"""30 normalized SRM (Spatial Rich Model) high-pass kernels.

Verbatim port of Fridrich & Kodovsky (2012) "Rich Models for Steganalysis
of Digital Images" via the official AIDE repository
(https://github.com/shilinyan99/AIDE/blob/main/models/srm_filter_kernel.py).

The kernels are 8 first-order direction (3x3), 4 second-order (3x3),
8 third-order (5x5), 4 edge-3x3, 4 edge-5x5, and 2 square (3x3, 5x5) =
30 total. They are pre-normalized by their characteristic coefficient so
that the L1 norm is comparable across kernels.
"""

from __future__ import annotations

import numpy as np


def _arr(rows: list[list[int]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


# 8 first-order direction filters (3x3): a unit at one offset minus the centre.
_class_1: list[np.ndarray] = [
    _arr([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    _arr([[0, 1, 0], [0, -1, 0], [0, 0, 0]]),
    _arr([[0, 0, 1], [0, -1, 0], [0, 0, 0]]),
    _arr([[0, 0, 0], [1, -1, 0], [0, 0, 0]]),
    _arr([[0, 0, 0], [0, -1, 1], [0, 0, 0]]),
    _arr([[0, 0, 0], [0, -1, 0], [1, 0, 0]]),
    _arr([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),
    _arr([[0, 0, 0], [0, -1, 0], [0, 0, 1]]),
]

# 4 second-order filters (3x3), normalized by /2.
_class_2: list[np.ndarray] = [
    _arr([[1, 0, 0], [0, -2, 0], [0, 0, 1]]) / 2,
    _arr([[0, 1, 0], [0, -2, 0], [0, 1, 0]]) / 2,
    _arr([[0, 0, 1], [0, -2, 0], [1, 0, 0]]) / 2,
    _arr([[0, 0, 0], [1, -2, 1], [0, 0, 0]]) / 2,
]

# 8 third-order filters (5x5), normalized by /3.
_class_3: list[np.ndarray] = [
    _arr([[-1, 0, 0, 0, 0], [0, 3, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]) / 3,
    _arr([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]) / 3,
    _arr([[0, 0, 0, 0, -1], [0, 0, 0, 3, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3,
    _arr([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3,
    _arr([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, -1]]) / 3,
    _arr([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0], [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]]) / 3,
    _arr([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -3, 0, 0], [0, 3, 0, 0, 0], [-1, 0, 0, 0, 0]]) / 3,
    _arr([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 3,
]

# 4 edge filters (3x3), normalized by /4.
_edge_3x3: list[np.ndarray] = [
    _arr([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]) / 4,
    _arr([[0, 2, -1], [0, -4, 2], [0, 2, -1]]) / 4,
    _arr([[0, 0, 0], [2, -4, 2], [-1, 2, -1]]) / 4,
    _arr([[-1, 2, 0], [2, -4, 0], [-1, 2, 0]]) / 4,
]

# 4 edge filters (5x5), normalized by /12.
_edge_5x5: list[np.ndarray] = [
    _arr([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 12,
    _arr([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2], [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]]) / 12,
    _arr([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12,
    _arr([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]]) / 12,
]

# Isotropic square filters, normalized by /4 (3x3) and /12 (5x5).
_square_3x3: np.ndarray = _arr([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4
_square_5x5: np.ndarray = (
    _arr([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]])
    / 12
)

all_normalized_hpf_list: list[np.ndarray] = (
    _class_1 + _class_2 + _class_3 + _edge_3x3 + _edge_5x5 + [_square_3x3, _square_5x5]
)
assert len(all_normalized_hpf_list) == 30
