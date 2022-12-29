from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List
from .backend_selection import array_api, NDArray


@dataclass
class SolverMetadata:
    n_iter: int
    norms: List[float]


class BaseSolver(ABC):
    def __init__(self):
        self.meta = None
    
    def solve(self, op, init_valie: NDArray) -> NDArray:
        self.meta = SolverMetadata(n_iter=0, norms=[])
        return self._solve(op, init_valie)

    @abstractmethod
    def _solve(self, op, init_valie: NDArray) -> NDArray:
        ...


class ForwardIteration(BaseSolver):
    def __init__(self, *, max_iter: int = 50, tol: float = 1e-4, eps: float = 1e-5):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def _solve(self, function: Callable[[NDArray], NDArray], init_value: NDArray) -> NDArray:
        prev_value, value = init_value, function(init_value)
        norm = None
        for i in range(self.max_iter):
            norm = array_api.norm(value - prev_value) / (self.eps + array_api.norm(value))
            self.meta.norms.append(norm.numpy()[0])

            if norm < self.tol:
                break

            prev_value, value = value, function(value)

        self.meta.n_iter = i
        return value
