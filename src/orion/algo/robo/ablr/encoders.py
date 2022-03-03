""" Module for the 'feature maps' from the paper. """
from abc import ABC, abstractmethod
from typing import Dict
import torch
import numpy as np
from torch import nn, Tensor
from torch.nn.utils import parameters_to_vector
from torch.nn import functional as F


class Encoder(nn.Module, ABC):
    """ Base class for an Encoder that maps samples from a given input space
    to vectors of length `out_features`.
    """

    def __init__(self, input_space: Dict, out_features: int):
        super().__init__()
        self.input_space: Dict = input_space
        self.in_features: int = len(input_space)
        self.out_features: int = out_features

    def parameter_vector(self) -> Tensor:
        """ Return a vector of all the weights of this model. """
        # This might come in handy when trying to replicate the paper's method.
        return parameters_to_vector(self.parameters())

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        pass


class NeuralNetEncoder(Encoder):
    """ Neural net encoder, in the style of DNGO or ABLR. """

    def __init__(self, input_space: Dict, out_features: int, hidden_neurons: int = 50):
        super().__init__(input_space=input_space, out_features=out_features)
        self.hidden_neurons = hidden_neurons
        self.dense = nn.Sequential(
            nn.Linear(self.in_features, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, self.out_features),
            nn.Tanh(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dense(inputs)


class AdaptiveEncoder(Encoder):
    """ Encoder with the same structure as the neural net encoder above, but
    where the weights of the first layer are learned and shared accross tasks on
    a per-hyper-parameter basis, using the Registry.
    
    For example, say that the registry currently contains information about the
    "learning_rate" and "momentum" hyper-parameters, and that we want to get an
    Encoder for a new Task with the following input space:
    ```
    {"lr": "log_uniform(1e-9, 1e-2)", "dropout_prob": "uniform(0., 0.8)"}
    ```
    
    This Encoder's first layer would have its weight matrix's first column
    loaded from the registry, such that we can already "encode" a given value of
    learning rate into a vector. The second column of the weight matrix would be
    randomly initialized and added to the registry such that it could be reused
    by following tasks.
    """

    def __init__(
        self,
        input_space: Dict,
        out_features: int,
        hidden_neurons: int = 50,
        registry: "Registry" = None,
    ):
        super().__init__(input_space=input_space, out_features=out_features)
        from .registry import Registry

        # TODO: Need to not store this registry on `self`, otherwise this causes
        # recursion problems in the __str__ of the modules that use the registry.
        registry = registry or Registry()
        self.hidden_neurons = hidden_neurons

        self.fc1_weight_columns: List[nn.Parameter] = nn.ParameterList()
        for i, hparam in enumerate(input_space):
            # Get the 'key' for this hyper-parameter.
            hparam_id: str = hparam
            weight_column = registry.get_hparam_code(
                hparam=hparam,
                hparam_space=input_space[hparam],
                feature_dims=self.hidden_neurons,
            )
            self.fc1_weight_columns.append(weight_column)
        # BUG: Saving this parameter on `self` causes an error when trying to deepcopy
        # this model. For now I'll just create it every time in the forward pass.
        # self.fc1_weight =  torch.hstack(tuple(self.fc1_weight_columns))
        # NOTE: Letting the bias be "free" on a per-task level, for now.
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_neurons))
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, out_features)

    def forward(self, x):
        # Compared to the DNGO encoder, we replace this:
        # x = torch.tanh(self.fc1(x))
        # With this:
        fc1_weight = torch.hstack(tuple(self.fc1_weight_columns))
        x = torch.tanh(F.linear(x, weight=fc1_weight, bias=self.fc1_bias))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class RandomFourierBasisEncoder(Encoder):
    """Random Fourier Basis Encoder, a.k.a. "Random Kitchen Sink encoder".
    
    Used as an alternative to the neural network encoder. Briefly described in
    section 4.1 of the ABLR paper. 
    """

    def __init__(
        self, input_space: Dict, out_features: int, kernel_bandwidth: float = 1.0
    ):
        super().__init__(input_space=input_space, out_features=out_features)
        p = self.in_features
        d = self.out_features
        # The "bandwidth of the approximated radial basis function kernel".
        self.kernel_bandwidth = nn.Parameter(torch.Tensor([kernel_bandwidth]))
        # TODO: Should these be re-sampled for each forward pass?
        self.U = torch.randn([d, p])
        self.b = torch.rand([d]) * 2 * np.pi
        # Store the constants, so we don't recompute them all the time.
        self.c1 = np.sqrt(2 / self.out_features)

    def forward(self, inputs: Tensor) -> Tensor:
        c2 = 1 / self.kernel_bandwidth
        return self.c1 * torch.cos(c2 * inputs @ self.U.T + self.b)
