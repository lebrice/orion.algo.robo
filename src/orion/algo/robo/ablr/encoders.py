""" Module for the 'feature maps' from the paper. """
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector


class Encoder(nn.Module, ABC):
    """Base class for an Encoder that maps samples from a given input space
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
