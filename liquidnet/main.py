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

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )

        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            w_reduced_synapse = torch.sum(w_activation, dim=1)
            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation
            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )
            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)
            v_pre = v_pre + 0.1 * f_prime
        
        return v_pre
    
    def _ode_step_runge_kutta(self, inputs, state):
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)
        
            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state
    
    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)
    
    def _f_prime(self, inputs, state): 
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(
            inputs, self.sensory_mu, self.sensory_sigma
        )
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)

            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = (
                torch.sum(sensory_in, dim=1)
                - v_pre * w_reduced_synapse
                + torch.sum(synapse_in, dim=1)
                - v_pre * w_reduced_sensory
            )

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime
        
        return f_prime
    
    def get_param_constrain_op(self):
        cm_clipping_op = torch.clamp(
            self.cm_t, self._cm_t_min_value, self._cm_t_max_value
        )
        gleak_clipping_op = torch.clamp(
            self.gleak, self._gleak_min_value, self._gleak_max_value
        )
        w_clipping_op = torch.clamp(self.W, self._w_min_value, self._w_max_value)
        sensory_w_clipping_op = torch.clamp(
            self.sensory_W, self._w_min_value, self._w_max_value
        )

        return [cm_clipping_op, gleak_clipping_op, w_clipping_op, sensory_w_clipping_op]
    
    def export_weights(self, dirname, output_weights=None):
        os.makedirs(dirname, exist_ok=True)
        w, erev, mu, sigma = (
            self.W.data.cpu().numpy(),
            self.erev.data.cpu().numpy(),
            self.mu.data.cpu().numpy(),
            self.sigma.data.cpu().numpy(),
        )
        sensory_w, sensory_erev, sensory_mu, sensory_sigma = (
            self.sensory_W.data.cpu().numpy(),
            self.sensory_erev.data.cpu().numpy(),
            self.sensory_mu.data.cpu().numpy(),
            self.sensory_sigma.data.cpu().numpy(),
        )
        vleak, gleak, cm = (
            self.vleak.data.cpu().numpy(),
            self.gleak.data.cpu().numpy(),
            self.cm_t.data.cpu().numpy()
        )

        if output_weights is not None:
            output_w, output_b = output_weights
            np.savetxt(
                os.path.join(dirname, "output_w.csv"), output_w.data.cpu().numpy()
            )
            np.savetxt(
                os.path.join(dirname, "output_b.csv"), output_b.data.cpu().numpy()
            )

        np.savetxt(os.path.join(dirname, "w.csv"), w)
        np.savetxt(os.path.join(dirname, "erev.csv"), erev)
        np.savetxt(os.path.join(dirname, "mu.csv"), mu)
        np.savetxt(os.path.join(dirname, "sigma.csv"), sigma)
        np.savetxt(os.path.join(dirname, "sensory_w.csv"), sensory_w)
        np.savetxt(os.path.join(dirname, "sensory_erev.csv"), sensory_erev)
        np.savetxt(os.path.join(dirname, "sensory_mu.csv"), sensory_mu)
        np.savetxt(os.path.join(dirname, "sensory_sigma.csv"), sensory_sigma)
        np.savetxt(os.path.join(dirname, "vleak.csv"), vleak)
        np.savetxt(os.path.join(dirname, "gleak.csv"), gleak)
        np.savetxt(os.path.join(dirname, "cm.csv"), cm)