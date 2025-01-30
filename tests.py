import os

import pytest
import torch
from torch import nn

from liquidnet.main import LiquidNet, MappingType, ODESolver
from liquidnet.vision import VisionLiquidNet

def test_vision_liquid_net_initialization():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)
    assert isinstance(model, nn.Module)
    assert isinstance(model.liquid_net, LiquidNet)

def test_vision_liquid_net_forward_pass():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)

    batch_size = 8
    channels = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    output = model(input_tensor)

    assert output.shape == (batch_size, num_classes)

def test_hidden_state_initialization():
    num_units = 64
    num_classes = 10
    model = VisionLiquidNet(num_units, num_classes)

    batch_size = 8
    channels = 3
    height = 32
    width = 32
    input_tensor = torch.randn(batch_size, channels, height, width)

    model(input_tensor)

    assert model.hidden_state is not None

@pytest.fixture
def vision_liquid_net():
    return VisionLiquidNet(num_units=64, num_classes=10)

@pytest.fixture
def liquid_net():
    return LiquidNet(num_units=64)

def test_vision_liquid_net_forward(vision_liquid_net):
    batch_size = 4
    input_channels = 3
    input_height = 32
    input_width = 32
    num_classes = 10

    inputs = torch.randn(batch_size, input_channels, input_height, input_width)

    outputs = vision_liquid_net(inputs)

    assert outputs.shape == (batch_size, num_classes)

def test_vision_liquid_net_hidden_state(vision_liquid_net):
    batch_size = 4
    input_channels = 3
    input_height = 32
    input_width = 32

    inputs = torch.randn(batch_size, input_channels, input_height, input_width)

    assert vision_liquid_net.hidden_state is None

    _ = vision_liquid_net(inputs)

    assert vision_liquid_net.hidden_state is not None

def test_liquid_net_forward(liquid_net):
    batch_size = 4
    input_size = 32
    num_units = liquid_net.state_size

    inputs = torch.randn(batch_size, input_size)
    initial_state = torch.zeros(batch_size, num_units)

    outputs, final_state = liquid_net(inputs, initial_state)

    assert outputs.shape == (batch_size, num_units)
    assert final_state.shape == (batch_size, num_units)

def test_liquid_net_parameter_constraints(liquid_net):
    constraints = liquid_net.get_param_constrain_op()
    for param in constraints:
        assert (param >= 0).all()

NUM_UNITS = 64
BATCH_SIZE = 4
INPUT_SIZE = 32
NUM_ITERATIONS = 100

@pytest.fixture
def liquid_net():
    return LiquidNet(NUM_UNITS)

@pytest.fixture
def sample_inputs():
    return torch.randn(BATCH_SIZE, INPUT_SIZE)

@pytest.fixture
def initial_state():
    return torch.zeros(BATCH_SIZE, NUM_UNITS)

def test_liquid_net_initialization(liquid_net):
    assert liquid_net.state_size == NUM_UNITS
    assert liquid_net.output_size == NUM_UNITS

def test_forward_pass(liquid_net, sample_inputs, initial_state):
    outputs, final_state = liquid_net(sample_inputs, initial_state)
    assert outputs.shape == (BATCH_SIZE, NUM_UNITS)
    assert final_state.shape == (BATCH_SIZE, NUM_UNITS)

def test_variable_constraints(liquid_net):
    constraining_ops = liquid_net.get_param_constrain_op()
    for op in constraining_ops:
        assert torch.all(op >= 0)

def test_export_weights(liquid_net):
    dirname = "test_weights"
    liquid_net.export_weights(dirname)

    assert os.path.exists(os.path.join(dirname, "w.csv"))
    assert os.path.exists(os.path.join(dirname, "erev.csv"))
    assert os.path.exists(os.path.join(dirname, "mu.csv"))
    assert os.path.exists(os.path.join(dirname, "sigma.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_w.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_erev.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_mu.csv"))
    assert os.path.exists(os.path.join(dirname, "sensory_sigma.csv"))
    assert os.path.exists(os.path.join(dirname, "vleak.csv"))
    assert os.path.exists(os.path.join(dirname, "gleak.csv"))
    assert os.path.exists(os.path.join(dirname, "cm.csv"))

@pytest.mark.parametrize("solver", [ODESolver.SemiImplicit, ODESolver.Explicit])
@pytest.mark.parametrize(
    "mapping_type", [MappingType.Identity, MappingType.Linear, MappingType.Affine]
)
def test_solver_and_mapping_types(
    liquid_net, sample_inputs, initial_state, solver, mapping_type
):
    liquid_net._solver = solver
    liquid_net._input_mapping = mapping_type
    outputs, final_state = liquid_net(sample_inputs, initial_state)

if __name__ == "__main__":
    pytest.main()