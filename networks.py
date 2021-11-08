import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
from typing import List
import numpy as np

###################33
### General network

def hidden_init(layer):
    inputs = layer.weight.data.size()[0]
    lim = 1./np.sqrt(inputs)
    return (-lim, lim)
    
class Network_with_hidden_layers_connected_to_input(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8],
                 activation_function=F.relu,
                 out_activation_function=None,
                 output_rescaling_function=None
                ):
        """Creates a multilayer perceptron neural network with relu activation (and softmax output)
        An additional connection from input to every layer is created. 
        These connections prevents diminishing gradient for deeper networks

        param: int input_size: length of the input vector
        param: int output_size: length of the output vector
        param: int seed: random seed
        param: List[int] hidden_layer_neurons: list of integers for the neural network hidden layers
        return: PyTorch network
        """
        super().__init__()
        self.activation_function = activation_function
        self.out_activation_function = out_activation_function
        self.output_rescaling_function = output_rescaling_function
        self.seed = torch.manual_seed(seed)
        
        inputs = [0]+hidden_layer_neurons[:-1]
        outputs = hidden_layer_neurons
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            new_layer = nn.Linear(inp+input_size, outp)
            self.fully_connected_layers.append(new_layer)
        for layer in self.fully_connected_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.fully_connected_layers.append(nn.Linear(outp+input_size,output_size))
        self.fully_connected_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fully_connected_layers[0](state))
        for layer in self.fully_connected_layers[1:-1]:
            x = self.activation_function(layer(torch.cat([state,x],1)))
        x = self.fully_connected_layers[-1](torch.cat([state,x],1))
        if self.out_activation_function:
            x = self.out_activation_function(x)
        if self.output_rescaling_function:
            x = self.output_rescaling_function(x)
        return x
    

class Actor_resnet(Network_with_hidden_layers_connected_to_input):
    def __init__(self,
                 state_size: int,
                 output_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8],
                 activation_function=F.relu,
                 out_activation_function=F.tanh,
                 output_rescaling_function=None
                ):
        super().__init__(
            input_size=state_size,
            output_size=output_size,
            seed=seed,
            hidden_layer_neurons=hidden_layer_neurons,
            activation_function=activation_function,
            out_activation_function=out_activation_function,
            output_rescaling_function=output_rescaling_function
        )

class Critic_resnet(Network_with_hidden_layers_connected_to_input):
    def __init__(self,
                 state_size: int,
                 action_size: int, 
                 output_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8],
                 activation_function=F.relu,
                 out_activation_function=F.tanh,
                 output_rescaling_function=None
                ):
        super().__init__(
            input_size=state_size+action_size,
            output_size=output_size,
            seed=seed,
            hidden_layer_neurons=hidden_layer_neurons,
            activation_function=activation_function,
            out_activation_function=out_activation_function,
            output_rescaling_function=output_rescaling_function
        )
    def forward(self, state, action):
        x = super().forward(torch.cat([state, action],dim=1))
        return x
    
###################33
### General network

class Actor(nn.Module):
    def __init__(self,
                 state_size: int,
                 output_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8],
                 activation_function=F.relu,
                 out_activation_function=F.tanh,
                 output_rescaling_function=None
                ):
        """Creates a multilayer perceptron neural network with relu activation (and softmax output)
        An additional connection from input to every layer is created. 
        These connections prevents diminishing gradient for deeper networks

        param: int input_size: length of the input vector
        param: int output_size: length of the output vector
        param: int seed: random seed
        param: List[int] hidden_layer_neurons: list of integers for the neural network hidden layers
        return: PyTorch network
        """
        super().__init__()
        self.activation_function = activation_function
        self.out_activation_function = out_activation_function
        self.output_rescaling_function = output_rescaling_function
        self.seed = torch.manual_seed(seed)
        
        inputs = [state_size]+hidden_layer_neurons[:-1]
        outputs = hidden_layer_neurons
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            new_layer = nn.Linear(inp, outp)
            self.fully_connected_layers.append(new_layer)
        self.fully_connected_layers.append(nn.Linear(outp, output_size))

    def forward(self, state):
        """Build a network that maps state -> action."""
        x = self.activation_function(self.fully_connected_layers[0](state))
        for layer in self.fully_connected_layers[1:-1]:
            x = self.activation_function(layer(x))
        x = self.fully_connected_layers[-1](x)
        if self.out_activation_function:
            x = self.out_activation_function(x)
        if self.output_rescaling_function:
            x = self.output_rescaling_function(x)
        return x

    
###################33
### General network

class Critic(nn.Module):
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 output_size: int,
                 seed: int,
                 hidden_layer_neurons: List[int]=[32,16,8],
                 activation_function=F.relu,
                 out_activation_function=F.tanh,
                 output_rescaling_function=None
                ):
        """Creates a multilayer perceptron neural network with relu activation (and softmax output)
        An additional connection from input to every layer is created. 
        These connections prevents diminishing gradient for deeper networks

        param: int input_size: length of the input vector
        param: int output_size: length of the output vector
        param: int seed: random seed
        param: List[int] hidden_layer_neurons: list of integers for the neural network hidden layers
        return: PyTorch network
        """
        super().__init__()
        self.activation_function = activation_function
        self.out_activation_function = out_activation_function
        self.output_rescaling_function = output_rescaling_function
        self.seed = torch.manual_seed(seed)
        
        inputs = [state_size]+[hidden_layer_neurons[0]+action_size] + hidden_layer_neurons[1:-1]
        outputs = hidden_layer_neurons
        self.fully_connected_layers = nn.ModuleList()
        for inp,outp in zip(inputs,outputs):
            new_layer = nn.Linear(inp, outp)
            self.fully_connected_layers.append(new_layer)
        self.fully_connected_layers.append(nn.Linear(outp, output_size))

    def forward(self, state, action):
        """Build a network that maps state -> action."""
        x = F.relu(self.fully_connected_layers[0](state))
        x = self.activation_function(self.fully_connected_layers[1](torch.cat([x, action], dim=1)))
        for layer in self.fully_connected_layers[2:-1]:
            x = self.activation_function(layer(x))
        x = self.fully_connected_layers[-1](x)
        if self.out_activation_function:
            x = self.out_activation_function(x)
        if self.output_rescaling_function:
            x = self.output_rescaling_function(x)
        return x
    
class Actor_batchn(nn.Module):
    def __init__(self,
                 state_size: int,
                 output_size: int,
                 **kwargs):
        super().__init__()
        self.state_size = state_size
        self.action_size = output_size
        self.layer_1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, output_size)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2))
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = F.relu(self.bn1(self.layer_1(states)))
        x = F.relu(self.layer_2(x))
        return torch.tanh(self.layer_3(x))

class Critic_batchn(nn.Module):
    def __init__(self, 
                 state_size: int,
                 action_size: int,
                 **kwargs):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_fc = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.value_fc1 = nn.Linear(128 +action_size, 128)
        self.output_fc = nn.Linear(128,1)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.value_fc1.weight.data.uniform_(*hidden_init(self.value_fc1))
        self.output_fc.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, action):
        xs = F.leaky_relu(self.bn1(self.state_fc(states)))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.value_fc1(x))
        return self.output_fc(x)