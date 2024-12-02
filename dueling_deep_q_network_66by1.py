'''
This code is mainly based on https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code.
'''
 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, num_rel_positions=2, save_load_file='~/'):
        super(DuelingDeepQNetwork, self).__init__()
        
        print('input dims = ' + str(input_dims))
        self.save_load_file = save_load_file

        fc_input_dims = input_dims[0]
        print('fc input dims = ' + str(fc_input_dims))
        print('#####' + str(fc_input_dims) + '#####')

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, 128)
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, n_actions)

        self.relu = nn.ReLU()

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')#('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = self.relu(self.fc1(state))
        flat2 = self.relu(self.fc2(flat1))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.save_load_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.save_load_file))
