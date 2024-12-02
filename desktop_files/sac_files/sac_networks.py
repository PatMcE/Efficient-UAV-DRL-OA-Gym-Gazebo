import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, action_size, device, hidden_size=32, num_rel_positions=2, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(ActorNetwork, self).__init__()
        self.device = device
        
        self.num_rel_positions = num_rel_positions
        self.init_w = init_w
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.image_input_dims = (input_dims[0] - 1,) + input_dims[1:]
        self.conv1 = nn.Conv2d(self.image_input_dims[0], 32, (4,4), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (2,2), stride=1)
        fc_input_dims = self.calculate_conv_output_dims(self.image_input_dims)
        print('#####' + str(fc_input_dims) + '#####')
        
        self.fc1 = nn.Linear(fc_input_dims, hidden_size)
        self.fc2 = nn.Linear(hidden_size+self.num_rel_positions, hidden_size+self.num_rel_positions)
        
        self.mu = nn.Linear(hidden_size+self.num_rel_positions, action_size)
        self.log_std_linear = nn.Linear(hidden_size+self.num_rel_positions, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)

    def calculate_conv_output_dims(self, image_input_dims):
        state = torch.zeros(1, *image_input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        #print('Actor state = ' + str(state.shape)) #Actor state = torch.Size([3, 80, 106])
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        image_state = state[:,0:-1,:,:]
        
        conv1 = F.relu(self.conv1(image_state))
        conv2 = F.relu(self.conv2(conv1))
        conv_state = conv2.view(conv2.size()[0], -1)
        
        flat1 = F.relu(self.fc1(conv_state), inplace=True)
        
        xyz_state = torch.stack([state[:,-1,0,0], state[:,-1,0,1]], dim=-1).float()
        cat = torch.cat((flat1, xyz_state), dim=1)        
        
        flat2 = F.relu(self.fc2(cat), inplace=True)
        mu = self.mu(flat2)

        log_std = self.log_std_linear(flat2)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon) 
        log_prob = log_prob.sum(1, keepdim=True) #new

        return action, log_prob            
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(self.device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, action_size, device, hidden_size=32, num_rel_positions=2, init_w=3e-3):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.num_rel_positions = num_rel_positions
        self.init_w = init_w
        
        self.image_input_dims = (input_dims[0] - 1,) + input_dims[1:]
        self.conv1 = nn.Conv2d(self.image_input_dims[0], 32, (4,4), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (2,2), stride=1)
        fc_input_dims = self.calculate_conv_output_dims(self.image_input_dims)
        print('#####' + str(fc_input_dims) + '#####')
        
        self.fc1 = nn.Linear(fc_input_dims+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size+self.num_rel_positions, hidden_size+self.num_rel_positions)
        self.fc3 = nn.Linear(hidden_size+self.num_rel_positions, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-self.init_w, self.init_w)

    def calculate_conv_output_dims(self, image_input_dims):
        state = torch.zeros(1, *image_input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        image_state = state[:,0:-1,:,:]
        
        conv1 = F.relu(self.conv1(image_state))
        conv2 = F.relu(self.conv2(conv1))
        conv_state = conv2.view(conv2.size()[0], -1)
        
        combined = torch.cat((conv_state, action), dim=1)
        flat1 = F.relu(self.fc1(combined), inplace=True)
        
        xyz_state = torch.stack([state[:,-1,0,0], state[:,-1,0,1]], dim=-1).float()
        cat = torch.cat((flat1, xyz_state), dim=1)        
        
        flat2 = F.relu(self.fc2(cat), inplace=True)     
        
        return self.fc3(flat2)
