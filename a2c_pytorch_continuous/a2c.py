import torch
from torch import nn
import torch.nn.functional as F

class A2C_Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, use_gpu, hidden_size=128):
        super(A2C_Actor, self).__init__()

        self.num_actions = num_actions

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.fc_mean = nn.Linear(hidden_size, num_actions)
        self.fc_log_std = nn.Linear(hidden_size, num_actions)

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.to(self.device)  # Move network to the device

    def forward(self, x): # Produces mean and log standard deviation of the action distribution
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
 
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  
        std = torch.exp(log_std)
        
        return mean, std

    def sample(self, state): # Samples an action space from the stochastic policy using the reparameterization trick
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) # Selected action

        log_prob = normal.log_prob(z) # Calculates the log probability of the acton 
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)  
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob 
    
    # def forward(self, state):
    #     policy_dist = F.relu(self.actor_linear1(state))
    #     policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

    #     return policy_dist
    
class A2C_Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, use_gpu, hidden_size=128):
        super(A2C_Critic, self).__init__()

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.to(self.device)  # Move network to the device
    
    def forward(self, state):
        #state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Move state to the correct device 
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        return value