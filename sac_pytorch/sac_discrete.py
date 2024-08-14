import torch
from torch import nn
import torch.nn.functional as F

#scales the cost function to encourage exploration
# smooth


# Actor Network
class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, use_gpu):
        super(SAC_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc_logits = nn.Linear(256, action_dim)
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)  # Output the raw logits for each action, the actual values from the layer
        probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
        return probs

    def sample(self, state):
        probs = self.forward(state)
        print(f'probs: {probs}')
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        print(f'action: {action}')
        log_prob = dist.log_prob(action)
        return action, log_prob


# Critic Network (Twin Critic Networks)
class SAC_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=256, hidden_dim_2=256):
        super(SAC_Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim_1)  # +1 for discrete action
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim_1)  # +1 for discrete action
        self.fc5 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc6 = nn.Linear(hidden_dim_2, 1)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, state, action):
        # state is expected to be of shape (batch_size, state_dim)
        # action is expected to be of shape (batch_size, 1) and contain integers
        # Convert action to one-hot encoding
        one_hot_action = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()
        
        x = torch.cat([state, one_hot_action], dim=1).to(self.device)

        # Q1 value
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2 value
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def Q1(self, state, action):
        # Convert action to one-hot encoding
        one_hot_action = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()
        
        x = torch.cat([state, one_hot_action], dim=1).to(self.device)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
    

# Value Network
class SAC_Entropy(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu):
        super(SAC_Entropy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


    
if __name__ == '__main__':
    state_dim = 12                                          # define number of input variables (the state)
    action_dim = 2                                          # define the number of possible outputs (the action)
    action_high = [1]
    action_low = [-1]                                      # define the maximum action value (continuous action space)
    net = SAC_Actor(state_dim, action_dim, action_high, action_low)      # define the network with the state dimensions (12) and action dimensions (2)
    state = torch.randn(1, state_dim)                       # create some random input
    output = net(state)                                     # send some random input into the network
    print(output)                                           # print the output