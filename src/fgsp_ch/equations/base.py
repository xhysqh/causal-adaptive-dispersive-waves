from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


class MomentumEquation(Protocol):
    """Interface required by a momentum-space time integrator."""

    def momentum_rhs(
        self, state: NDArray[np.floating]
    ) -> NDArray[np.floating]: ...

    def state_rhs(
        self, state: NDArray[np.floating]
    ) -> NDArray[np.floating]: ...

    def linearized_transport(
        self, coefficient_state: NDArray[np.floating]
    ) -> sparse.csr_matrix: ...

