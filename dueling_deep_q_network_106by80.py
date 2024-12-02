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
        self.image_input_dims = (input_dims[0] - 1,) + input_dims[1:]
        
        self.conv1 = nn.Conv2d(self.image_input_dims[0], 32, (8,8), stride=4)#10,14
        self.conv2 = nn.Conv2d(32, 64, (4,4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1)

        fc_input_dims = self.calculate_conv_output_dims(self.image_input_dims)
        print('#####' + str(fc_input_dims) + '#####')

        self.fc1 = nn.Linear(fc_input_dims, 1024)
        self.fc2 = nn.Linear(1024+num_rel_positions, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, n_actions)

        self.relu = nn.ReLU()

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')#('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, image_input_dims):
        state = T.zeros(1, *image_input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        image_state = state[:,0:-1,:,:]
        
        conv1 = self.relu(self.conv1(image_state))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        flat1 = self.relu(self.fc1(conv_state))
        
        xyz_state = T.stack([state[:,-1,0,0], state[:,-1,0,1]], dim=-1).float()
        cat = T.cat((flat1, xyz_state), dim=1)
        flat2 = self.relu(self.fc2(cat))

        V = self.V(flat2)
        A = self.A(flat2)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.save_load_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.save_load_file))
