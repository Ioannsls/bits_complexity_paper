import jax.numpy as jnp
import numpy as np
from jax.numpy import ndarray
import abc
from typing import Callable


class ICompressor(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def compressor_function(self, x) -> ndarray:
        pass

    def get_compressor_func(self) -> Callable:
        return self.compressor_function


class Topk(ICompressor):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def compressor_function(self, x):
        sorted_indices = jnp.flip(jnp.argsort(jnp.abs(x)))  # Sort indices
        result = jnp.zeros_like(x)
        result = result.at[sorted_indices[:self.k]].set(
            x[sorted_indices[:self.k]])
        return result
    
class RandK(ICompressor):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def compressor_function(self, x) -> ndarray:
        choice_indexes = np.random.choice(jnp.arange(len(x)), self.k, replace= False)
        results = np.zeros(len(x))
        results[choice_indexes] = x[choice_indexes]
        return jnp.array(results)
        


class UnbiasedExp(ICompressor):  # TODO rewrite in vector operation
    def __init__(self, sequence: ndarray) -> None:
        super().__init__()
        self.sequence = sequence

    def compressor_function(self, x):
        res_vector = ndarray([])
        idx_in_sequence = jnp.searchsorted(
            self.sequence, jnp.abs(x), side='right')
        for i in list(zip(jnp.abs(x),
                          self.sequence[idx_in_sequence - 1],
                          self.sequence[idx_in_sequence])):
            p1 = (i[2] - i[0]) / (-i[1] + i[2])
            res_vector = jnp.append(res_vector, jnp.random.choice(
                [i[1], i[2]], p=[p1, 1 - p1]))
        return jnp.sign(x) * res_vector


class BiasedExp(ICompressor):
    def __init__(self, sequence: ndarray) -> None:
        super().__init__()
        self.sequence = sequence

    def compressor_function(self, x) -> ndarray:
        return jnp.sign(
            x) * jnp.min(jnp.abs(self.sequence - jnp.abs(x)), axis=0)


class ExpDithering(ICompressor):  # TODO normalize or check is x normalize
    def __init__(self, p, b, u) -> None:
        super().__init__()
        self.p = p
        self.b = b
        self.u = u

    def ksi_random_variable(self, x) -> ndarray:  # TODO rewrite in vector operator
        ksi = jnp.array([])
        values = jnp.array([[self.b ** (-self.u - 1), self.b ** (-self.u)]])
        for t in x:
            p1 = (self.b ** (-self.u) - t) / \
                (self.b ** (-self.u) - self.b ** (-self.u - 1))
            ksi = jnp.append(
                arr=ksi, values=jnp.random.choice(
                    values, p=[
                        p1, 1 - p1]))
        return ksi

    def compressor_function(self, x) -> ndarray:
        result = jnp.linalg.norm(x, self.p) * jnp.sign(x)
        return result * \
            self.ksi_random_variable(jnp.abs(x) / jnp.linalg.norm(x, self.p))
