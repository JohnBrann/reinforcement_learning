import torch
from torch import nn
import torch.nn.functional as F

from tkinter import Variable

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, use_gpu, hidden_size=256):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

        # if use_gpu and torch.cuda.is_available():
        #     self.device = 'cuda'
        # else:
        #     self.device = 'cpu'
    
    def forward(self, state):
       # state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        state = torch.FloatTensor(state).unsqueeze(0)#.to(self.device)

        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
    