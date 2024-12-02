'''
This code is mainly based on https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code.
'''
 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, num_rel_positions=2, save_load_file='~/', pretrained_path=None):
        super(DuelingDeepQNetwork, self).__init__()
        
        # Store the path for saving/loading the model
        self.save_load_file = save_load_file
        self.image_input_dims = (input_dims[0] - 1,) + input_dims[1:]

        # Define convolutional layers
        self.conv1 = nn.Conv2d(self.image_input_dims[0], 32, (4,4), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (2,2), stride=1)

        # Calculate fully connected input dimensions
        fc_input_dims = self.calculate_conv_output_dims(self.image_input_dims)
        print('#####' + str(fc_input_dims) + '#####')

        # Define fully connected layers (ensure these are defined in __init__)
        self.fc1 = nn.Linear(fc_input_dims, 1024)  # First fully connected layer
        self.fc2 = nn.Linear(1024 + num_rel_positions, 256)  # Second fully connected layer
        self.V = nn.Linear(256, 1)  # Value stream
        self.A = nn.Linear(256, n_actions)  # Advantage stream

        # Optimizer and loss initialization should come after setting up layers
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        # Initialize device before loading pre-trained convs
        self.device = T.device('cpu')  # Use 'cuda:0' if GPU is available
        self.to(self.device)

        # Load pre-trained convolutional layers if a path is provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
            print('loading pretrained weights')

        # Freeze convolutional layers (optional)
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = False

        # Relu activation function
        self.relu = nn.ReLU()

    def _load_pretrained(self, pretrained_path):
        print(f'Loading pre-trained weights from {pretrained_path}')
        pretrained_dict = T.load(pretrained_path, map_location=self.device) # Load pre-trained model

        # Get the current model's state_dict (parameters)
        model_dict = self.state_dict()

        # Filter out layers from the pre-trained model that exist in the current model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update the current model's state_dict with the pre-trained weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        # Check if the weights were loaded correctly by comparing layer names
        loaded_layers = [k for k in pretrained_dict.keys()]
        print(f'Successfully loaded layers: {loaded_layers}')

    def calculate_conv_output_dims(self, image_input_dims):
        """Calculate dimensions after convolutional layers."""
        state = T.zeros(1, *image_input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        """Forward pass through the network."""
        image_state = state[:, 0:-1, :, :]
        
        conv1 = self.relu(self.conv1(image_state))
        conv2 = self.relu(self.conv2(conv1))
        conv_state = conv2.view(conv2.size()[0], -1)
        flat1 = self.relu(self.fc1(conv_state))
        
        xyz_state = T.stack([state[:, -1, 0, 0], state[:, -1, 0, 1]], dim=-1).float()
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
