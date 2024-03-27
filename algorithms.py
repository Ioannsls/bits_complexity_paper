import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from tqdm import tqdm
from dataclasses import dataclass
import abc
from typing import Any, Callable


debug_mode = True

######################################################################


class DynamicDataTypeImitation():
    def __init__(self, bits_per_variable: int = 4) -> None:
        self.rounding_set = None
        self.bits_per_variable = bits_per_variable
        self.rounding_set_lengtn = 2**(self.bits_per_variable - 1)
        self.shift_max_val = self.rounding_set_lengtn // 10
        self.shift_min_val = self.rounding_set_lengtn // 10

    def set_shift_max_val(self, new_shift: int):
        self.shift_max_val = new_shift

    def set_shift_min_val(self, new_shift: int):
        self.shift_min_val = new_shift

    def _set_update_condition(self, compressed_data: ndarray) -> bool:
        if(self.rounding_set is None):
            return True
        min_value = jnp.min(jnp.abs(compressed_data))
        argmin_index = jnp.argmin(jnp.abs(self.rounding_set - min_value))
        if (argmin_index < self.shift_min_val):
            return True
        if(argmin_index > self.rounding_set_lengtn - self.shift_min_val):
            print("something in DDT not ok")
            return True
        return False

    def _update_rounding_set(self, compressed_data: ndarray) -> None:
        if(self._set_update_condition(compressed_data)):
            max_value = jnp.max(jnp.abs(compressed_data))
            max_degree = jnp.ceil(jnp.log2(max_value)) + self.shift_max_val
            rounding_set_degrees = jnp.flip(
                jnp.cumsum(-jnp.ones(self.rounding_set_lengtn)) + 1) + max_degree
            self.rounding_set = 2**rounding_set_degrees

    def dynamic_data_type(self, compressed_data: ndarray) -> ndarray:
        if(debug_mode):
            assert(self.rounding_set is not None), "rounding set is not initialize"
            assert(jnp.max(self.rounding_set) !=
                   0), "maximum of rounding set == 0"
        matr_rounding_set, matr_data = jnp.broadcast_arrays(
            self.rounding_set, jnp.reshape(
                jnp.abs(compressed_data), [
                    len(compressed_data), 1]))
        return jnp.sign(compressed_data) * \
            self.rounding_set[jnp.argmin(matr_rounding_set - matr_data, axis=1)]

    def clear_infos(self):
        self.rounding_set = None

######################################################################


@dataclass
class AlgoLoggerParams():
    is_logging_node_val: bool = False
    is_logging_master_val: bool = False
    node_x_applied_func: list[Callable[[ndarray], ndarray]] = None
    master_x_applied_func: list[Callable[[ndarray], ndarray]] = None
    logging_rate: float = 0


class AlgoLogger():
    def __init__(self,
                 params: AlgoLoggerParams,
                 file_name: str,
                 ) -> None:
        self.params = params
        self.file_name = file_name
        # use vars(class object instance)
        # как-то дернуть ифну про окружение из под которого запускается, инфу
        # про сам алгоритм и его параметры, что мы хотим собирать
        pass

    def node_logger(self):
        if(self.params.is_logging_node_val):
            pass
        pass

    def master_logger(self):
        if(self.params.is_logging_master_val):
            pass
        pass


######################################################################
class Node():
    def __init__(
            self,
            node_step: Callable[..., None],
            func_in_node: Callable[[ndarray], ndarray],
            compressor: Callable[[ndarray], ndarray],
            x: ndarray) -> None:
        self.step = node_step
        self.func = func_in_node
        self.grad_func = jax.grad(self.func, 0)
        self.compressor = compressor
        self.x = x

    def compute(self) -> ndarray:
        self.step(self)


class Master():
    def __init__(
            self,
            nodes_quantity: int,
            master_step: Callable[..., None],
            x: ndarray) -> None:
        self.nodes_list = []
        self.nodes_quantity = nodes_quantity
        self.step = master_step
        self.x = x

    def compute(self) -> None:
        self.step(self)


class IAlgorithm(abc.ABC):
    def __init__(
            self,
            starting_point: ndarray,
            learning_rate: float,
            iteration_number: int,
            compressor_func: Callable[[ndarray], ndarray],
            nodes_quantity: int,
            node_func: list[Callable[[ndarray], ndarray]]) -> None:
        super().__init__()
        self.startion_point = starting_point
        self.learning_rate = learning_rate
        self.iteration_number = iteration_number
        self.comressor_func = compressor_func
        self.nodes_quantity = nodes_quantity
        self.node_func = node_func

    @abc.abstractmethod
    def _node_step(node) -> None:
        pass

    @abc.abstractmethod
    def _master_step(master) -> None:
        pass

    @abc.abstractmethod
    def _init_set_param_master(self) -> None:
        pass

    @abc.abstractmethod
    def _init_set_param_nodes(self) -> None:
        pass

    @abc.abstractmethod
    def _alg_step(self) -> None:
        pass

    def run_algo(self):
        self._init_set_param_master()
        self._init_set_param_nodes()
        for _ in tqdm(range(self.iteration_number)):
            self._alg_step()


class EF21Node(Node):
    def __init__(self, node_step: Callable[..., None], func_in_node: Callable[[
                 ndarray], ndarray], compressor: Callable[[ndarray], ndarray], x: ndarray) -> None:
        super().__init__(node_step, func_in_node, compressor, x)
        self.c = 0
        self.g = 0


class EF21Master(Master):
    def __init__(self,
                 nodes_quantity: int,
                 master_step: Callable[...,
                                       None],
                 x: ndarray) -> None:
        super().__init__(nodes_quantity, master_step, x)
        self.learning_rate = None
        self.g = 0


class EF21(IAlgorithm):
    def __init__(self,
                 starting_point: jax.Array,
                 learning_rate: float,
                 iteration_number: int,
                 compressor_func: Callable[...,
                                           Any],
                 nodes_quantity: int,
                 node_func: list[Callable[...,
                                          Any]]) -> None:
        super().__init__(
            starting_point,
            learning_rate,
            iteration_number,
            compressor_func,
            nodes_quantity,
            node_func)

    def _node_step(self, node: EF21Node):
        node.c = node.compressor(node.grad_func(node.x) - node.g)
        node.g += node.c

    def _master_step(self, master: EF21Master):
        def send_x_to_nodes(nodes_list: list[EF21Node]) -> None:
            for node in nodes_list:
                node.x = master.x

        def all_node_compute(nodes_list: list[EF21Node]):
            for node in nodes_list:
                node.compute()

        def collect_mean_c_from_nodes(nodes_list: list[EF21Node]):
            return jnp.mean(jnp.array([node.c for node in nodes_list]), axis=0)

        master.x -= master.learning_rate * master.g
        send_x_to_nodes(master.nodes_list)
        all_node_compute(master.nodes_list)
        master.g += collect_mean_c_from_nodes(master.nodes_list)

    def _alg_step(self):
        self.master.compute()
        # collect info about alg

    def _init_set_param_master(self) -> None:
        self.master = EF21Master(
            self.nodes_quantity,
            self._master_step,
            self.startion_point)
        self.master.learning_rate = self.learning_rate

    def _init_set_param_nodes(self) -> None:
        def init_nodes() -> list:
            if (debug_mode):
                assert self.nodes_quantity == len(
                    self.node_func), "nodes_quantity != len(node_func)"
            return [
                EF21Node(
                    self._node_step,
                    self.node_func[i],
                    self.comressor_func,
                    self.startion_point) for i in range(
                    self.nodes_quantity)]

        self.master.nodes_list = init_nodes()


'''
class MARINA():
    def __init__(self, iteration_number, step_size, compressor_name) -> None:
        super().__init__(iteration_number, step_size, compressor_name, "MARINA")
        self.inf_on_step = np.array([])

    def run_algo(
            self,
            func_in_nodes,
            probability,
            nodes_amount,
            start_point=np.ones(d)):
        self.nodes = [
            Node(
                func_in_nodes[i],
                start_point,
                Node.gradient(
                    func_in_nodes[i],
                    start_point),
                self.compressor) for i in range(nodes_amount)]
        average_grad = [self.nodes[i].g for i in range(nodes_amount)]
        self.master = Master(
            self.nodes,
            self.step_size,
            average_grad,
            self.alg_name,
            start_point,
            average_grad)  # создание мастера
        g = average_grad
        self.inf_on_step = jnp.append(g * nodes_amount, self.inf_on_step)
        for _ in tqdm(range(self.iter_amount)):
            g = self.alg_step(g)

    def alg_step(self, g):
        values = self.master.master_compute(g)
        node_values = [
            self.nodes[i].compute(
                values, "MARINA") for i in range(
                self.nodes.size)]  # Тут непонятен формат values
        return node_values

    def marina_step(self, parameters_for_compressor,
                    parameters_for_marina: dict) -> ndarray:
        x_next = self.x - \
            parameters_for_marina["step"] * parameters_for_marina["g"]
        g = self.gradient(self.x) if parameters_for_marina["c"] == 1 else self.compressor(
            self.gradient(x_next) - self.gradient(self.x), *parameters_for_compressor)
        self.x = x_next
        return g

    def marina_master(self, nodes_values: ndarray, possib) -> ndarray:
        c = np.random.choice([1, 0], p=[possib, 1 - possib])  # ????
        return ndarray(buffer=np.array(c, np.mean(nodes_values, axis=0)))
'''
