# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""An implementation of quantum neural network regressor."""
from __future__ import annotations

from typing import Callable, cast

import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import Optimizer
from qiskit.opflow import OperatorBase
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.utils import QuantumInstance

from .neural_network_regressor import NeuralNetworkRegressor
from ...deprecation import warn_deprecated, DeprecatedType
from ...neural_networks import TwoLayerQNN, EstimatorQNN
from ...utils import derive_num_qubits_feature_map_ansatz
from ...utils.loss_functions import Loss


class VQR(NeuralNetworkRegressor):
    """Quantum neural network regressor."""

    def __init__(
        self,
        num_qubits: int | None = None,
        feature_map: QuantumCircuit | None = None,
        ansatz: QuantumCircuit | None = None,
        observable: QuantumCircuit | OperatorBase | BaseOperator | None = None,
        loss: str | Loss = "squared_error",
        optimizer: Optimizer | None = None,
        warm_start: bool = False,
        quantum_instance: QuantumInstance | None = None,
        initial_point: np.ndarray | None = None,
        callback: Callable[[np.ndarray, float], None] | None = None,
        *,
        estimator: BaseEstimator | None = None,
    ) -> None:
        r"""
        Args:
            num_qubits: The number of qubits for the underlying QNN.
                If ``None`` is given, the number of qubits is derived from the
                feature map or ansatz. If neither of those is given, raises an exception.
                The number of qubits in the feature map and ansatz are adjusted to this
                number if required.
            feature_map: The (parametrized) circuit to be used as a feature map for the underlying
                QNN. If ``None`` is given, the :class:`~qiskit.circuit.library.ZZFeatureMap`
                is used if the number of qubits is larger than 1. For a single qubit regression
                problem the :class:`~qiskit.circuit.library.ZFeatureMap` is used by default.
            ansatz: The (parametrized) circuit to be used as an ansatz for the underlying
                QNN. If ``None`` is given then the :class:`~qiskit.circuit.library.RealAmplitudes`
                circuit is used.
            observable: The observable to be measured in the underlying QNN. If ``None``,
                use the default :math:`Z^{\otimes num\_qubits}` observable.
            loss: A target loss function to be used in training. Default is squared error.
            optimizer: An instance of an optimizer to be used in training. When ``None`` defaults
                to SLSQP.
            warm_start: Use weights from previous fit to start next fit.
            quantum_instance: If a quantum instance is set and ``estimator`` is ``None``,
                the underlying QNN will be of type
                :class:`~qiskit_machine_learning.neural_networks.TwoLayerQNN`, and the quantum
                instance will be used to compute the neural network's results. If an estimator
                instance is also set, it will override the `quantum_instance` parameter and
                a :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`
                will be used instead.
            initial_point: Initial point for the optimizer to start from.
            callback: a reference to a user's callback function that has two parameters and
                returns ``None``. The callback can access intermediate data during training.
                On each iteration an optimizer invokes the callback and passes current weights
                as an array and a computed value as a float of the objective function being
                optimized. This allows to track how well optimization / training process is going on.
            estimator: If an estimator instance is set, the underlying QNN will be of type
                :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`, and the estimator
                primitive will be used to compute the neural network's results.
        Raises:
            QiskitMachineLearningError: Needs at least one out of ``num_qubits``, ``feature_map`` or
                ``ansatz`` to be given. Or the number of qubits in the feature map and/or ansatz
                can't be adjusted to ``num_qubits``.
        """
        # needed for mypy
        neural_network: EstimatorQNN | TwoLayerQNN = None
        if quantum_instance is not None and estimator is None:
            warn_deprecated(
                "0.5.0", DeprecatedType.ARGUMENT, old_name="quantum_instance", new_name="estimator"
            )
            self._quantum_instance = quantum_instance
            self._estimator = None

            # construct QNN
            neural_network = TwoLayerQNN(
                num_qubits=num_qubits,
                feature_map=feature_map,
                ansatz=ansatz,
                observable=observable,
                quantum_instance=quantum_instance,
                input_gradients=False,
            )
        else:
            # construct estimator QNN by default
            self._quantum_instance = None
            self._estimator = estimator

            num_qubits, feature_map, ansatz = derive_num_qubits_feature_map_ansatz(
                num_qubits, feature_map, ansatz
            )

            # construct circuit
            self._feature_map = feature_map
            self._ansatz = ansatz
            self._num_qubits = num_qubits
            circuit = QuantumCircuit(self._num_qubits)
            circuit.compose(self._feature_map, inplace=True)
            circuit.compose(self._ansatz, inplace=True)

            # construct observable
            if observable is None:
                observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
            self._observable = observable

            neural_network = EstimatorQNN(
                estimator=estimator,
                circuit=circuit,
                observables=[self._observable],
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters,
            )

        super().__init__(
            neural_network=neural_network,
            loss=loss,
            optimizer=optimizer,
            warm_start=warm_start,
            initial_point=initial_point,
            callback=callback,
        )

    @property
    def feature_map(self) -> QuantumCircuit:
        """Returns the used feature map."""
        if self._quantum_instance is not None and self._estimator is None:
            return cast(TwoLayerQNN, super().neural_network).feature_map
        else:
            return self._feature_map

    @property
    def ansatz(self) -> QuantumCircuit:
        """Returns the used ansatz."""
        if self._quantum_instance is not None and self._estimator is None:
            return cast(TwoLayerQNN, super().neural_network).ansatz
        else:
            return self._ansatz

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits used by ansatz and feature map."""
        if self._quantum_instance is not None and self._estimator is None:
            return cast(TwoLayerQNN, super().neural_network).num_qubits
        else:
            return self._num_qubits
