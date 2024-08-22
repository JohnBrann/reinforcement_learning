import torch
from torch import nn
import torch.nn.functional as F

#scales the cost function to encourage exploration
# smooth


# Actor Network
class SAC_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, dims=256):
        super(SAC_Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, dims)
        self.fc2 = nn.Linear(dims, dims)
        self.fc_logits = nn.Linear(dims, action_dim)
        
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)  # Raw logits
        probs = F.softmax(logits, dim=-1)  # Stable softmax

        print(probs)
        probs = torch.clamp(probs, min=1e-8)  # Avoid zero probabilities
        return probs

    def sample_nondeterministic(self, state):
        probs = self.forward(state)
      

        # # Normalize (though softmax already does this)
        # probs_sum = probs.sum(dim=1, keepdim=True)
        # probs_sum = torch.clamp(probs_sum, min=1e-8)  # Avoid division by zero
        # probs = probs / probs_sum

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob

    def get_action_distributions(self, state):
        action_probs = self.forward(state)  # action_probs will have shape [256, 2]
        log_probs = torch.log(action_probs)  # log_probs will have shape [256, 2]
        return action_probs, log_probs

     # This samples an action determinsitically, selecting the most probable action
    # def sample_deterministic(self, state):
    #     action_probs = self.forward(state)
    #     action_probs = action_probs.squeeze()
    #     action = torch.argmax(action_probs).item()
    #     log_prob = torch.log(action_probs[action])
    #     return action, log_prob

    def sample_deterministic(self, state):
        action_probs = self.forward(state)
        action = torch.argmax(action_probs, dim=-1)  # action will have shape [256]
        log_prob = torch.log(action_probs[range(action_probs.size(0)), action])  # log_prob will have shape [256]
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

        # # Q2 architecture
        # self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim_1)  # +1 for discrete action
        # self.fc5 = nn.Linear(hidden_dim_1, hidden_dim_2)
        # self.fc6 = nn.Linear(hidden_dim_2, 2)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, state, action):
        # state is expected to be of shape (batch_size, state_dim)
        # action is expected to be of shape (batch_size, 1) and contain integers

        # If action is not one-hot encoded already, convert it
        action = action.argmax(dim=-1)  # This line may not be needed if actions are already correct
        one_hot_action = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()

        # Concatenate state and action tensors
        x = torch.cat([state, one_hot_action], dim=1).to(self.device)

        # Forward pass through the network layers
        q = self.fc1(x)  # First fully connected layer
        q = F.relu(q)    # ReLU activation, no in-place modification
        q = self.fc2(q)  # Second fully connected layer
        q = F.relu(q)    # ReLU activation, again no in-place modification
        q = self.fc3(q)  # Final layer output

        return q

    def Q1(self, state, action):
        # Convert action to one-hot encoding
        one_hot_action = F.one_hot(action.long().squeeze(-1), num_classes=self.action_dim).float()
        
        x = torch.cat([state, one_hot_action], dim=1).to(self.device)

        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1


class SAC_Value(nn.Module):
    def __init__(self, state_dim, action_dim, use_gpu, hidden_dim_1=256, hidden_dim_2=256):
        super(SAC_Value, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q1 architecture
        self.fc1 = nn.Linear(self.state_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1,  hidden_dim_2)
        self.v = nn.Linear(hidden_dim_2, 1)

        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
 


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
