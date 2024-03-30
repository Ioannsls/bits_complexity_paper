import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from tqdm import tqdm
from dataclasses import dataclass
import abc
from typing import Callable, Type, Any
import problems
import compressors
import logging
from csv_logger import CsvLogger
from os import remove, path
debug_mode = True

######################################################################


class DynamicDataTypeImitation():
    def __init__(self, bits_per_variable: int = 8) -> None:
        self.rounding_set = None
        self.bits_per_variable = bits_per_variable
        self.rounding_set_lengtn = 2**(self.bits_per_variable - 1)
        self.max_val_shift_on_create = 0
        self.max_val_shift_on_conditions = 1 + self.rounding_set_lengtn // 5

    def set_shift_on_create(self, new_shift: int):
        self.max_val_shift_on_create = new_shift

    def set_shift_on_conditions(self, new_shift: int):
        self.max_val_shift_on_conditions = new_shift

    def _set_update_condition(self, compressed_data: ndarray) -> bool:
        if(self.rounding_set is None):
            return True
        max_value = jnp.max(jnp.abs(compressed_data))
        argmax_index = jnp.argmin(jnp.abs(self.rounding_set - max_value))
        if (argmax_index + 1 < self.rounding_set_lengtn -
                self.max_val_shift_on_conditions):
            return True
        return False

    def check_update_rounding_set(self, compressed_data: ndarray) -> None:
        if(self._set_update_condition(compressed_data)):
            max_value = jnp.max(jnp.abs(compressed_data))
            max_degree = jnp.ceil(jnp.log2(max_value)) + \
                self.max_val_shift_on_create
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
        return jnp.sign(compressed_data) * self.rounding_set[jnp.argmin(
            jnp.abs(matr_rounding_set - matr_data), axis=1)]

    def clear_info(self):
        self.rounding_set = None

######################################################################


@dataclass
class AlgoLoggerParams():
    is_logging_iteration: bool = True
    is_logging_bits_complexity: bool = False
    is_logging_time: bool = False
    is_logging_node_val = False
    node_info_applied_func: list[Callable[[ndarray], Any]] = None
    is_logging_master_val = False
    master_info_applied_func: list[Callable[[ndarray], Any]] = None
    logging_rate: int = 10


class AlgoLogger():
    interested_info_about_algo = [
        'problem_class_object',
        'learning_rate',
        'iteration_number',
        'comressor_class_object',
        'nodes_quantity']

    def __init__(self,
                 params: AlgoLoggerParams,
                 file_name: str) -> None:
        self.params = params
        self.file_name = file_name
        self.master_log_info = None
        self.node_log_info = None
        self.bits_log = 0
        self.iteration_number = 0
        if(path.exists(file_name)):
            remove(file_name)

        if(self.params.master_info_applied_func is not None):
            self.params.is_logging_master_val = True
        if(self.params.node_info_applied_func is not None):
            self.params.is_logging_node_val = True

        self._new_log_set()

        def get_csv_header() -> list:
            header = []
            if(self.params.is_logging_iteration):
                header.append('iteration')
            if(self.params.is_logging_bits_complexity):
                header.append('bits_complexity')
            if(self.params.is_logging_master_val):
                for func in self.params.master_info_applied_func:
                    header.append(func.__name__)
            if(self.params.is_logging_node_val):
                for func in self.params.node_info_applied_func:
                    header.append(func.__name__)
            return header

        def get_csv_logger_format() -> str:
            delimiter = ','
            if (self.params.is_logging_time):
                return f'%(asctime)s{delimiter}%(message)s'
            return f'%(message)s'
        self.csv_logger = CsvLogger(
            filename=self.file_name,
            level=logging.INFO,
            fmt=get_csv_logger_format(),
            datefmt='%H:%M:%S:%f',
            header=get_csv_header())

    def set_info_about_alg(self, algo) -> None:
        self.algo_class_object = algo
        alg_info = []
        algo_dict = self.algo_class_object.__dict__
        for name in self.interested_info_about_algo:
            adding_str = name + ' = ' + str(algo_dict[name])
            alg_info.append(adding_str)
        self.info_about_alg = alg_info
        self.csv_logger.info(alg_info)

    def _new_log_set(self):
        if(self.params.is_logging_master_val):
            self.master_log_info = [None] * \
                len(self.params.master_info_applied_func)
        if(self.params.is_logging_node_val):
            self.node_log_info = [[] for _ in range(
                len(self.params.node_info_applied_func))]

    def _is_log_now(self) -> bool:
        if(self.iteration_number % self.params.logging_rate == 0):
            return True
        return False

    def add_node_log(self, point=None):
        if(self.params.is_logging_node_val and self._is_log_now()):
            assert(point is not None), "point == None"
            funcs = self.params.node_info_applied_func
            for i in range(len(funcs)):
                self.node_log_info[i].append(funcs[i](point))

    def add_master_log(self, point=None):
        if(self.params.is_logging_master_val and self._is_log_now()):
            assert(point is not None), "point == None"
            funcs = self.params.master_info_applied_func
            for i in range(len(funcs)):
                self.master_log_info[i] = funcs[i](point)

    def add_bits_complexity_log(self, bits_complexity: int = 0) -> None:
        if(self.params.is_logging_bits_complexity):
            self.bits_log += bits_complexity

    def log_info(self) -> None:
        self.iteration_number += 1
        if(self._is_log_now()):
            msg = []
            if(self.params.is_logging_iteration):
                msg.append(self.iteration_number)
            if(self.params.is_logging_bits_complexity):
                msg.append(self.bits_log)
            if(self.params.is_logging_master_val):
                msg.extend(self.master_log_info)
            if(self.params.is_logging_node_val):
                msg.extend(self.node_log_info)
            self.csv_logger.info(msg)
            self._new_log_set()


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
        self.nodes_list: Type[Node] = []
        self.nodes_quantity = nodes_quantity
        self.step = master_step
        self.x = x

    def compute(self) -> None:
        self.step(self)


class IAlgorithm(abc.ABC):
    def __init__(
            self,
            problem_class_object: Type[problems.IProblem],
            learning_rate: float,
            iteration_number: int,
            compressor_class_object: Type[compressors.ICompressor],
            nodes_quantity: int,
            DDT: DynamicDataTypeImitation = None,
            logger: AlgoLogger = None) -> None:
        super().__init__()
        self.problem_class_object = problem_class_object
        self.startion_point = self.problem_class_object.get_start_point()
        self.learning_rate = learning_rate
        self.iteration_number = iteration_number
        self.comressor_class_object = compressor_class_object
        self.comressor_func = compressor_class_object.get_compressor_func()
        self.nodes_quantity = nodes_quantity
        self.node_func = self.problem_class_object.get_func_list(
            nodes_quantity)
        self.ddt = DDT
        self.logger = logger
        self.ddt_is_init = (DDT is not None)
        self.logger_is_inti = (logger is not None)

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
        if(self.logger_is_inti):
            self.logger.set_info_about_alg(self)
        if(self.ddt_is_init):
            self.ddt.check_update_rounding_set(self.startion_point + 1)
        for _ in tqdm(range(self.iteration_number)):
            self._alg_step()


class EF21Node(Node):
    def __init__(self, node_step: Callable[..., None], func_in_node: Callable[[
                 ndarray], ndarray], compressor: Callable[[ndarray], ndarray], x: ndarray) -> None:
        super().__init__(node_step, func_in_node, compressor, x)
        self.c = 0
        self.g = 0  # self.compressor(self.grad_func(self.x))


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
    def __init__(
            self,
            problem_class_object: problems.IProblem,
            learning_rate: float,
            iteration_number: int,
            compressor_class_object: compressors.ICompressor,
            nodes_quantity: int,
            DDT: DynamicDataTypeImitation = None,
            logger: AlgoLogger = None) -> None:
        super().__init__(
            problem_class_object,
            learning_rate,
            iteration_number,
            compressor_class_object,
            nodes_quantity,
            DDT,
            logger)
        self.steps = 0

    def _node_step(self, node: EF21Node):
        node.c = node.compressor(node.grad_func(node.x) - node.g)
        if(self.logger_is_inti):
            self.logger.add_node_log(node.c)
        if(self.ddt_is_init):
            node.c = self.ddt.dynamic_data_type(node.c)
        node.g += node.c

    def _master_step(self, master: EF21Master):
        def send_x_to_nodes(nodes_list: list[EF21Node]) -> None:
            for node in nodes_list:
                node.x = master.x

        def all_node_compute(nodes_list: list[EF21Node]) -> None:
            for node in nodes_list:
                node.compute()

        def collect_mean_c_from_nodes(nodes_list: list[EF21Node]) -> ndarray:
            if(self.logger_is_inti):
                if(self.ddt_is_init):
                    self.logger.add_bits_complexity_log(
                        self.nodes_quantity * self.ddt.bits_per_variable)
                else:
                    self.logger.add_bits_complexity_log(
                        self.nodes_quantity * 32)
            return jnp.mean(jnp.array([node.c for node in nodes_list]), axis=0)

        master.x -= master.learning_rate * master.g
        send_x_to_nodes(master.nodes_list)
        all_node_compute(master.nodes_list)
        collected_c = collect_mean_c_from_nodes(master.nodes_list)
        master.g += collected_c
        if(self.ddt_is_init):
            self.ddt.check_update_rounding_set(collected_c)
        if(self.logger_is_inti):
            self.logger.add_master_log(master.x)

    def _alg_step(self):
        self.master.compute()
        if(self.logger_is_inti):
            self.logger.log_info()

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

        def master_g(node_list: list[EF21Node]):
            node_g = []
            for node in node_list:
                node_g.append(node.g)
            return jnp.mean(jnp.array(node_g))
        #self.master.g = master_g(self.master.nodes_list)
