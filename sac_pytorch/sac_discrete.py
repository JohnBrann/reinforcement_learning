import torch
from torch import nn
import torch.nn.functional as F

#scales the cost function to encourage exploration
# smooth


# Actor Network
class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, use_gpu, dims=256):
        super(SAC_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, dims)
        self.fc2 = nn.Linear(dims,dims)
        self.fc_logits = nn.Linear(dims, action_dim)
        
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

    # This samples an action non-determinsitically, selecting a random action from the distribution
    def sample_nondeterministic(self, state):
        probs = self.forward(state)
        #print(f'probs: {probs}')
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        #print(f'action: {action}')
        log_prob = dist.log_prob(action)
        return action, log_prob

     # This samples an action determinsitically, selecting the most probable action
    def sample_deterministic(self, state):
        action_probs = self.forward(state)
        action_probs = action_probs.squeeze()
        action = torch.argmax(action_probs).item()
        log_prob = torch.log(action_probs[action])
        return action, log_prob


# Critic Network (Twin Critic Networks)
class SAC_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=250, hidden_dim_2=250):
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
        #action = action.unsqueeze(1)
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


def main():
    # Parameters
    state_dim = 4  # Example state dimension
    action_dim = 2  # Example action dimension (for discrete actions)
    action_low = -1  # Example action space lower bound
    action_high = 1  # Example action space upper bound
    use_gpu = False  # Set to True if you want to use GPU

    # Initialize actor and critic networks
    actor = SAC_Actor(state_dim, action_dim, action_low, action_high, use_gpu)
    critic = SAC_Critic(state_dim, action_dim, use_gpu)

    # Example input (batch of states and actions)
    batch_size = 12
    states = torch.randn(batch_size, state_dim)  # Random states
    actions = torch.randint(0, action_dim, (batch_size, 1))  # Random discrete actions

    # Test Actor Network
    actor_probs = actor(states)
    sampled_action, log_prob = actor.sample(states)
    print("Actor output probabilities:", actor_probs)
    print("Sampled action:", sampled_action)
    print("Log probability of the sampled action:", log_prob)

    # Test Critic Network
    q1, q2 = critic(states, actions)
    print("Critic Q1 output:", q1)
    print("Critic Q2 output:", q2)

    # Test Q1 function
    q1_value = critic.Q1(states, actions)
    print("Q1 function output:", q1_value)

if __name__ == "__main__":
    main()
