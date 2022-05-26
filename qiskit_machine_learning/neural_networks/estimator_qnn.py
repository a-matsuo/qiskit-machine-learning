import logging
from numbers import Integral
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit.primitives.gradient.base_estimator_gradient import \
    BaseEstimatorGradient
from qiskit_machine_learning.neural_networks import NeuralNetwork

logger = logging.getLogger(__name__)


class EstimatorQNN(NeuralNetwork):

    def __init__(
            self,
            estimator: Estimator,
            input_params: Optional[List[Parameter]] = None,
            weight_params: Optional[List[Parameter]] = None,
            gradient: BaseEstimatorGradient = None,
            input_gradients: bool = False,
    ):
        self._estimator = estimator
        self._input_params = list(input_params or [])
        self._weight_params = list(weight_params or [])
        self._gradient = gradient
        super().__init__(
            len(self._input_params),
            len(self._weight_params),
            sparse=False,
            output_shape=self._compute_output_shape(),
            input_gradients=input_gradients,
        )

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Determines the output shape of a given operator."""
        # TODO: supports multi-observable or not?
        # return (len(self._estimator.observables), )
        return (1,)

    def forward(
            self,
            input_data: Optional[Union[List[float], np.ndarray, float]],
            weights: Optional[Union[List[float], np.ndarray, float]],
    ) -> np.ndarray:
        result = self._forward(input_data, weights)
        return result

    def _forward(
            self, input_data: Optional[np.ndarray], weights: Optional[np.ndarray]
    ) -> np.ndarray:
        # evaluate operator
        num_samples = input_data.shape[0]
        #print("num samples:", num_samples)
        results = np.zeros((num_samples, *self.output_shape))

        for i in range(num_samples):
            param_values = [input_data[i, j] for j, _ in enumerate(self._input_params)]

            param_values += [weights[j] for j, _ in enumerate(self._weight_params)]

            #print("param values: ", param_values)
            for j, _ in enumerate(self._estimator.observables):
                result = self._estimator([0],[j], [param_values])
                results[i, j]=result.values
            #print("result: ", result)

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
        #print("num samples:", num_samples)
        # input_grad =
        # weights_grad =
        #results = np.zeros((num_samples, 1, self._num_inputs + self._num_weights))
        #prob = np.zeros((num_samples, *self._output_shape))

        input_grad = np.zeros((num_samples, 1, self._num_inputs)) if self._input_gradients else None

        weights_grad = np.zeros((num_samples, 1, self._num_weights))

        #print('input_grad: ',input_grad)

        for i in range(num_samples):
            param_values = [input_data[i, j] for j, input_param in enumerate(self._input_params)]

            param_values += [weights[j] for j, weight_param in enumerate(self._weight_params)]

            #print("param values: ", param_values)

            if self._input_gradients:
                #compute gradients with respect to input data
                result = self._gradient.gradient(param_values)
                input_grad[i][0]=result.values[:self._num_inputs]

            else:
                #compute gradients for only weights
                result = self._gradient.gradient(param_values, partial=self._gradient._circuit.parameters[self._num_inputs:])

            weights_grad[i][0]=result.values[self._num_inputs:]
            #print("result: ", result)

        return input_grad, weights_grad
