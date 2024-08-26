import torch
from torch import nn
import torch.nn.functional as F

class A2C_Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, use_gpu, hidden_size=256):
        super(A2C_Actor, self).__init__()

        self.num_actions = num_actions

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        #self.to(self.device)  # Move network to the device
    
    def forward(self, state):
       # state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)


        return policy_dist
    
class A2C_Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, use_gpu, hidden_size=256):
        super(A2C_Critic, self).__init__()

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.to(self.device)  # Move network to the device
    
    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)#.unsqueeze(0).to(self.device)

        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        return value