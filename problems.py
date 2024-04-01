import abc
import jax.numpy as jnp
import numpy as np
from typing import Callable
from jax.numpy import ndarray
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


class IProblem(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def get_start_point(self) -> ndarray:
        pass

    @abc.abstractmethod
    def get_func_list(
            self, quantity_of_func: int) -> list[Callable[[ndarray], ndarray]]:
        pass


class MushroomsLogLos(IProblem):
    def __init__(self,
                 file_path: str = "datasets/",
                 file_name: str = "mushrooms.txt",
                 test_size: float = 0.2) -> None:
        assert test_size < 1, "test size >= 1"
        data = load_svmlight_file(file_path + file_name)
        self.X, self.Y = data[0].toarray(), data[1]
        self.X_shape = self.X.shape
        for i in range(self.X_shape[0]):
            if (self.Y[i] == 2):
                self.Y[i] = -1.0
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X, self.Y, test_size=test_size)

    def get_start_point(self) -> ndarray:
        return jnp.zeros(self.X_shape[1])

    @staticmethod
    def func(w: ndarray, x, y):
        column_q, row_q = x.shape
        return jnp.mean(jnp.log(jnp.ones(column_q) +
                        jnp.exp(-(x @ w.T) * y)).reshape(column_q, 1))

    def get_func_list(
            self, quantity_of_func: int) -> list[Callable[[ndarray], ndarray]]:
        X_split = jnp.array_split(self.X_train, quantity_of_func, axis=0)
        Y_split = jnp.array_split(self.Y_train, quantity_of_func)
        func_list = []
        for i in range(quantity_of_func):
            func_list.append(
                lambda w, i=i: MushroomsLogLos.func(
                    w, X_split[i], Y_split[i]))
        return func_list

    def precision_comp(self, w: ndarray) -> ndarray:
        tmp = (self.X_test @ w.T).T * self.Y_test
        if(len(jnp.shape(tmp)) == 1):
            return jnp.count_nonzero(tmp > 0) / len(self.X_test)
        return jnp.count_nonzero(tmp > 0, axis=1) / len(self.X_test)


class A9A(IProblem):
    funcs_name = {"LL", "NLLSQ"}

    def __init__(self,
                 file_path: str = "datasets/",
                 train_file_name: str = "a9a.txt",
                 test_file_name: str = "a9a_test.txt",
                 loss_func_name: str = "NLLSQ") -> None:
        train_data = load_svmlight_file(file_path + train_file_name)
        test_data = load_svmlight_file(file_path + test_file_name)
        self.X_test, self.Y_test = test_data[0].toarray(), test_data[1]
        self.X_train, self.Y_train = train_data[0].toarray(), train_data[1]
        self.X_test_shape = self.X_test.shape
        self.X_train_shape = self.X_train.shape
        assert {loss_func_name}.issubset(
            self.funcs_name), "func name not in funcs"
        if (loss_func_name == "LL"):
            pass
        elif (loss_func_name == "NLLSQ"):
            for i in range(self.X_test_shape[0]):
                if (self.Y_test[i] == -1):
                    self.Y_test[i] = 0
            for i in range(self.X_train_shape[0]):
                if (self.Y_train[i] == -1):
                    self.Y_train[i] = 0
        self.func_name = loss_func_name

    def get_start_point(self) -> ndarray:
        return jnp.zeros(self.X_train_shape[1])

    @staticmethod
    def log_loss_func(w: ndarray, x, y):
        column_q, row_q = x.shape
        return jnp.mean(jnp.log(jnp.ones(column_q) +
                        jnp.exp(-(x @ w.T) * y)).reshape(column_q, 1))

    @staticmethod
    def non_linear_least_squares_loss(w: ndarray, x, y):
        return jnp.mean((y - 1 / (1 + jnp.exp(-(x @ w.T))))**2)

    def func(self, w, x, y):
        if (self.func_name == "LL"):
            return(A9A.log_loss_func(w, x, y))
        elif (self.func_name == "NLLSQ"):
            return(A9A.non_linear_least_squares_loss(w, x, y))

    def get_func_list(
            self, quantity_of_func: int) -> list[Callable[[ndarray], ndarray]]:
        X_split = jnp.array_split(self.X_train, quantity_of_func, axis=0)
        Y_split = jnp.array_split(self.Y_train, quantity_of_func)
        func_list = []
        for i in range(quantity_of_func):
            func_list.append(
                lambda w, i=i: self.func(
                    w, X_split[i], Y_split[i]))
        return func_list

    def precision_comp(self, w: ndarray) -> ndarray:
        if (self.func_name == "LL"):
            tmp = (self.X_test @ w.T).T * self.Y_test
            if(len(jnp.shape(tmp)) == 1):
                return jnp.count_nonzero(tmp > 0) / len(self.X_test)
            return jnp.count_nonzero(tmp > 0, axis=1) / len(self.X_test)
        elif (self.func_name == "NLLSQ"):
            tmp = ((self.X_test @ w.T).T - 0.5) * (self.Y_test - 0.5)
            if(len(jnp.shape(tmp)) == 1):
                return jnp.count_nonzero(tmp < 0.5) / len(self.X_test)
            return jnp.count_nonzero(tmp < 0.5, axis=1) / len(self.X_test)
