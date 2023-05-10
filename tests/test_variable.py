import torch
import torch.nn as nn
import numpy as np
import tree
import tests.mocks as mocks
from clowder import variable


class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def dummy_network():
    return MLP([32, 50, 10])

def test_update():
    torch.manual_seed(1)
    input_tensor = torch.zeros(size=(1, 32))
    dummy_network_instance = dummy_network()
    params = list(dummy_network_instance.parameters())
    variable_source = mocks.MockVariableSource(params)
    variable_client = variable.VariableClient(variable_source, key="policy")
    variable_client.update_and_wait()
    tree.map_structure(torch.equal, variable_client.params, params)
    
def test_multiple_keys():
    torch.manual_seed(1)
    input_tensor = torch.zeros(size=(1, 32))
    dummy_network_instance = dummy_network()
    params = list(dummy_network_instance.parameters())
    steps = np.zeros(shape=1)
    variables = {'network': params, 'steps': steps}
    variable_source = mocks.MockVariableSource(variables, use_default_key=False)
    variable_client = variable.VariableClient(variable_source, key=['network', 'steps'])
    variable_client.update_and_wait()
    tree.map_structure(torch.equal, variable_client.params[0],
                       params)
    tree.map_structure(np.testing.assert_array_equal, variable_client.params[1],
                       steps)
    
if __name__ == "__main__":
    test_update()