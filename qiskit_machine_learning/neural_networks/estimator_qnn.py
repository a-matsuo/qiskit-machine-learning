import logging
from numbers import Integral
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives.gradient.base_estimator_gradient import BaseEstimatorGradient
#from .base_estimator_gradient import BaseEstimatorGradient

from qiskit_machine_learning.exceptions import (QiskitError,
                                                QiskitMachineLearningError)
from qiskit.primitives import Estimator


from scipy.sparse import coo_matrix


logger = logging.getLogger(__name__)


class EstimatorQNN():

    def __init__(
            self,
            estimator: Estimator,
            input_params: Optional[List[Parameter]] = None,
            weight_params: Optional[List[Parameter]] = None,
            gradient: BaseEstimatorGradient = None,
            input_gradients: bool = False,
    ):
        # IGNORING SPARSE
        # SKIPPING CUSTOM GRADIENT
        # SKIPPING "INPUT GRADIENTS" -> by default with primitives?

        self._estimator = estimator

        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])

        self._num_inputs = len(self._input_params)
        self._num_weights = len(self._weight_params)
        # the circuit must always have measurements.... (?)
        # add measurements in case none are given

        self._gradient = gradient
        self._input_gradients = input_gradients

    # def set_interpret(
    #     self,
    #     interpret: Optional[Callable[[int], Union[int, Tuple[int, ...]]]],
    #     output_shape: Union[int, Tuple[int, ...]] = None,
    # ) -> None:
    #     """Change 'interpret' and corresponding 'output_shape'. If self.sampling==True, the
    #     output _shape does not have to be set and is inferred from the interpret function.
    #     Otherwise, the output_shape needs to be given.

    #     Args:
    #         interpret: A callable that maps the measured integer to another unsigned integer or
    #             tuple of unsigned integers. See constructor for more details.
    #         output_shape: The output shape of the custom interpretation, only used in the case
    #             where an interpret function is provided and ``sampling==False``. See constructor
    #             for more details.
    #     """

    #     # save original values
    #     self._original_output_shape = output_shape
    #     self._original_interpret = interpret

    #     # derive target values to be used in computations
    #     self._output_shape = self._compute_output_shape(interpret, output_shape)
    #     self._interpret = interpret if interpret is not None else lambda x: x

    def _compute_output_shape(self, interpret, output_shape) -> Tuple[int, ...]:
        """Validate and compute the output shape."""

        # # this definition is required by mypy
        # output_shape_: Tuple[int, ...] = (-1,)
        # # todo: move sampling code to the super class

        # if interpret is not None:
        #     if output_shape is None:
        #         raise QiskitMachineLearningError(
        #             "No output shape given, but required in case of custom interpret!"
        #         )
        #     if isinstance(output_shape, Integral):
        #         output_shape = int(output_shape)
        #         output_shape_ = (output_shape,)
        #     else:
        #         output_shape_ = output_shape
        # else:
        #     if output_shape is not None:
        #         # Warn user that output_shape parameter will be ignored
        #         logger.warning(
        #             "No interpret function given, output_shape will be automatically "
        #             "determined as 2^num_qubits."
        #         )

        #     output_shape_ = (2 ** self._circuit.num_qubits,)

        # # final validation
        # output_shape_ = self._validate_output_shape(output_shape_)

        return (1,)

    # analogous to set quantum instance

    # def set_sampler(self, sampler):
    #     # no transpilation
    #     self._sampler = sampler

    # def set_gradient(self, gradient):
    #     # no transpilation
    #     self._gradient = gradient

    def forward(
            self,
            input_data: Optional[Union[List[float], np.ndarray, float]],
            weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> np.ndarray:

        print("input data:", input_data)
        print("weights data:", weights)
        result = self._forward(input_data, weights)
        return result

    def _forward(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        # evaluate operator
        num_samples = input_data.shape[0]
        print("num samples:", num_samples)
        results = np.zeros((num_samples, 1))
        #prob = np.zeros((num_samples, *self._output_shape))

        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]

            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]

            print("param values: ", param_values)
            result = self._estimator([0],[0], [param_values])
            results[i]=result.values
            print("result: ", result)

        return results


    def backward(
            self,
            input_data: Optional[Union[List[float], np.ndarray, float]],
            weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:

        result = self._backward(input_data, weights)
        return result

    def _backward(
        self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],]:
        # evaluate operator
        num_samples = input_data.shape[0]
        print("num samples:", num_samples)
        # input_grad =
        # weights_grad =
        #results = np.zeros((num_samples, 1, self._num_inputs + self._num_weights))
        #prob = np.zeros((num_samples, *self._output_shape))

        input_grad = np.zeros((num_samples, 1, self._num_inputs)) if self._input_gradients else None

        weights_grad = np.zeros((num_samples, 1, self._num_weights))

        print('input_grad: ',input_grad)

        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]

            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]

            print("param values: ", param_values)

            if self._input_gradients:
                #compute gradients with respect to input data
                result = self._gradient.gradient(param_values)
                input_grad[i][0]=result.values[:self._num_inputs]

            else:
                #compute gradients for only weights
                result = self._gradient.gradient(param_values, partial=self._gradient._circuit.parameters[self._num_inputs:])

            weights_grad[i][0]=result.values[self._num_inputs:]
            print("result: ", result)

        return input_grad, weights_grad
