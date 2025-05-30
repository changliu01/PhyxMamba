from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

class NBeatsBlock(torch.nn.Module):
    """
    N-BEATS block which takes a basis function as an argumentorch.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: torch.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecastorch.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [torch.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = torch.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(torch.nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self, blocks: torch.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        # residuals = x.flip(dims=(1,))
        residuals = x
        forecast = 0
        # input_mask = input_mask.flip(dims=(1,))
        # forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(torch.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(torch.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = torch.nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float32) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)
        self.forecast_time = torch.nn.Parameter(
            torch.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float32) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=torch.float32), requires_grad=False)

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = torch.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(torch.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = torch.nn.Parameter(torch.tensor(np.transpose(np.cos(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = torch.nn.Parameter(torch.tensor(np.transpose(np.sin(backcast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = torch.nn.Parameter(torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = torch.nn.Parameter(torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32),
                                                    requires_grad=False)

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = torch.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = torch.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast