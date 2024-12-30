import os
import numpy as np
import torch
import torch.nn as nn

from enum import Enum

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class LiquidNet(nn.Module):
    
    def __init__(self, num_units):
        super(LiquidNet, self).__init__()

        self._input_size = -1
        self._num_units = num_units
        self._is_built = False

        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit

        self._input_mapping = MappingType.Affine

        self._erev_init_factor = 1

        self._w_init_max = 1.0
        self._w_init_min = 0.01
        self._cm_init_max = 0.5
        self._cm_init_min = 0.5
        self._gleak_init_max = 1
        self._gleak_init_min = 1

        self._w_min_value = 0.00001
        self._w_max_value = 1000
        self._gleak_min_value = 0.00001
        self._gleak_max_value = 1000
        self._cm_t_min_value = 0.000001
        self._cm_t_max_value = 1000

        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None
    
    @property
    def state_size(self):
        return self._num_units
    
    @property
    def output_size(self):
        return self._num_units
    
    def _map_inputs(self, inputs, reuse_scope=False):
        if (
            self._input_mapping == MappingType.Affine
            or self._input_mapping == MappingType.Linear
        ):
            w = nn.Parameter(torch.ones(self._input_size))
            inputs = inputs * w

        if self._input_mapping == MappingType.Affine:
            b = nn.Parameter(torch.zeros(self._input_size))
            inputs = inputs + b
            
        return inputs
    
    def forward(self, inputs, state):
        if not self._is_built:
            self._is_built = True
            self._input_size = int(inputs.shape[-1])
            self._get_variables()
        
        elif self._input_size != int(inputs.shape[-1]):
            raise ValueError(
                "You first feed an input with {} features and now one with {} features, that is not possible".format(
                    self._input_size, int(inputs[-1])
                )
            )
        
        inputs = self._map_inputs(inputs)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(
                inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds
            )
        
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        
        else:
            raise ValueError("Unknown ODE Solver '{}'".format(str(self._solver)))
        
        outputs = next_state

        return outputs, next_state

    def _get_variables(self):
        self.sensory_mu = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 0.5 + 0.3
        )
        self.sensory_sigma = nn.Parameter(
            torch.rand(self._input_size, self._num_units) * 5.0 + 3.0
        )
        self.sensory_W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._input_size, self._num_units],
                )
            )
        )
        sensory_erev_init = (
            2 * np.random.randint(low=0, high=2, size=[self._input_size, self._num_units]) - 1
        )
        self.sensory_erev = nn.Parameter(
            torch.Tensor(sensory_erev_init * self._erev_init_factor)
        )
        self.mu = nn.Parameter(torch.rand(self._num_units, self._num_units) * 0.5 + 0.3)
        self.sigma = nn.Parameter(
            torch.rand(self._num_units, self._num_units) * 5.0 + 3.0
        )
        self.W = nn.Parameter(
            torch.Tensor(
                np.random.uniform(
                    low=self._w_init_min,
                    high=self._w_init_max,
                    size=[self._num_units, self._num_units],
                )
            )
        )
        erev_init = (
            2 * np.random.randint(low=0, high=2, size=[self._num_units, self._num_units]) - 1
        )
        self.erev = nn.Parameter(torch.Tensor(erev_init * self._erev_init_factor))

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self._num_units) * 0.4 - 0.2)
        else:
            self.vleak = nn.Parameter(torch.Tensor(self._fix_vleak))
        
        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                self.gleak = nn.Parameter(
                    torch.rand(self._num_units)
                    * (self._gleak_init_max - self._gleak_init_min)
                    + self._gleak_init_min
                )
            else:
                self.cm_t = nn.Parameter(
                    torch.Tensor([self._cm_init_min] * self._num_units)
                )
        else:
            self.cm_t = nn.Parameter(torch.Tensor(self._fix_cm))

    
    def _ode_step(self, inputs, state):
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator
        
        return v_pre

    
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)