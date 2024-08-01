from collections import deque
import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
import torch.nn.functional as F
import yaml

from td3 import TD3_Actor, TD3_Critic

from datetime import datetime, timedelta
import argparse
import itertools

import os

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def size(self):
        return len(self.buffer)

# TD3 Agent
class Agent():

    def __init__(self, is_training, endless, continue_training, render, use_gpu, hyperparameter_set):
        with open(os.path.join(os.getcwd(), 'hyperparameters.yml'), 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id                 = hyperparameters['env_id']
        self.state_dim              = hyperparameters['state_dim']
        self.action_dim             = hyperparameters['action_dim']
        self.action_low             = hyperparameters['action_low']
        self.action_high            = hyperparameters['action_high']
        self.replay_buffer_size     = hyperparameters['replay_memory_size']         # size of replay memory
        self.batch_size             = hyperparameters['mini_batch_size']            # size of the training data set sampled from the replay memory
        self.discount               = hyperparameters['discount']
        self.tau                    = hyperparameters['tau']
        self.learning_rate          = hyperparameters['learning_rate']
        self.policy_noise           = hyperparameters['policy_noise']
        self.noise_clip             = hyperparameters['noise_clip']
        self.policy_freq            = hyperparameters['policy_freq']
        self.model_save_freq        = hyperparameters['model_save_freq']
        self.max_reward             = hyperparameters['max_reward']
        self.max_timestep           = hyperparameters['max_timestep']
        self.max_episodes           = hyperparameters['max_episodes']
        self.env_make_params        = hyperparameters.get('env_make_params',{})     # Get optional environment-specific parameters, default to empty dict

        if continue_training:
            suffix = '_cont'
        else:
            suffix = ''

        # Path to Run info, create if does not exist
        self.RUNS_DIR = "runs"
        os.makedirs(self.RUNS_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(self.RUNS_DIR, f'{self.hyperparameter_set}{suffix}.log')
        self.MODEL_FILE = os.path.join(self.RUNS_DIR, f'{self.hyperparameter_set}{suffix}.pt')
        self.GRAPH_FILE = os.path.join(self.RUNS_DIR, f'{self.hyperparameter_set}{suffix}.png')
        self.DATE_FORMAT = "%m-%d %H:%M:%S"

        # Set device based on device arg
        if use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # set endless mode if endless arg is true, otherwise set max episodes based on parameters 
        if endless or not is_training:
            self.max_episodes = itertools.count()
        else:
            self.max_episodes = range(self.max_episodes)

        # Create instance of the environment.
        self.env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions & observation space size
        self.num_actions = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        self.rewards_per_episode = []
        self.total_it = 0

        # Create actor and critic networks
        self.actor = TD3_Actor(self.num_states, self.num_actions, use_gpu, self.action_low, self.action_high).to(self.device)
        self.actor_target = TD3_Actor(self.num_states, self.action_dim, use_gpu, self.action_low, self.action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = TD3_Critic(self.num_states, self.num_actions, use_gpu).to(self.device)
        self.critic_target = TD3_Critic(self.num_states, self.num_actions, use_gpu).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        if is_training:
            # Initialize log file
            start_time = datetime.now()
            self.last_graph_update_time = start_time

            log_message = f"{start_time.strftime(self.DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

            if continue_training:
                self.load(self.RUNS_DIR, f'{self.hyperparameter_set}')

        # if we are not training, generate the actor and critic policies based on the saved model
        else:
            self.load(self.RUNS_DIR, f'{self.hyperparameter_set}')
            self.actor.eval()
            self.critic.eval()
            start_time = datetime.now()
            log_message = f"{start_time.strftime(self.DATE_FORMAT)}: Run starting..."
            print(log_message)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).detach().cpu().numpy().flatten()
    
    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = torch.FloatTensor(actions).data.normal_(0, self.policy_noise).clamp(-self.noise_clip, self.noise_clip).to(self.device)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.action_high, self.action_high)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Compute current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    def run(self, is_training=True):

        best_reward = -np.inf   # Used to track best reward
        best_average_reward = -np.inf # Used to track best average reward across last 100 episodes

        for episode in self.max_episodes:

            state, _ = self.env.reset()  # Initialize environment. Reset returns (state,info).
            terminated = False      # True when agent reaches goal or fails
            truncated = False       # True when max_timestep is reached
            episode_reward = 0.0    # Used to accumulate rewards per episode
            step_count = 0          # Used for syncing policy => target network

            if not is_training:
                self.load(self.RUNS_DIR, f'{self.hyperparameter_set}')

            while(not terminated and not truncated and not step_count == self.max_timestep):
                action = self.select_action(state)
                if is_training:
                    noise = np.random.normal(0, self.action_high * self.policy_noise, size=self.action_dim)
                    action = (action + noise).clip(-self.action_high, self.action_high)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                terminated = step_count == self.max_timestep - 1 or terminated

                if is_training:
                    self.replay_buffer.add(state, action, reward, next_state, terminated)

                    if self.replay_buffer.size() > self.batch_size and step_count % self.policy_freq == 0:
                        # Train the agent
                        self.train()

                state = next_state
                episode_reward += reward
                step_count += 1

            # Keep track of the rewards collected per episode and save model
            self.rewards_per_episode.append(episode_reward)

            if is_training:
                current_time = datetime.now()
                if current_time - self.last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(self.rewards_per_episode)
                    self.last_graph_update_time = current_time
                
                if (episode + 1) % 100 == 0:
                    average_reward = np.mean(self.rewards_per_episode[-100:])
                    time_now = datetime.now()
                    log_message = f"{time_now.strftime(self.DATE_FORMAT)}: Average Reward over last 100 episodes: {average_reward:0.1f} at episode: {episode + 1}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    if average_reward > best_average_reward:
                        best_average_reward = average_reward  # Update the best average reward
                        # Save model
                        self.save(self.RUNS_DIR, f'{self.hyperparameter_set}')
                        log_message = f"{time_now.strftime(self.DATE_FORMAT)}: New Best Average Reward: {best_average_reward:0.1f} at episode: {episode + 1}, saving model..."
                        print(log_message)
                        with open(self.LOG_FILE, 'a') as file:
                            file.write(log_message + '\n')
        
                if episode_reward > best_reward and episode > 0:
                    # Print message
                    best_reward = episode_reward
                    log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: New Best Reward: {episode_reward:0.1f} ({abs((episode_reward-best_reward)/best_reward)*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
            else:
                log_message = f"{datetime.now().strftime(self.DATE_FORMAT)}: This Episode Reward: {episode_reward:0.1f}"
                print(log_message)

    # There is no functional difference between . pt and . pth when saving PyTorch models
    def save(self, directory, name):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), f"{directory}/{name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{name}_critic.pth")

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load(f"{directory}/{name}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{name}_critic.pth"))

    def save_graph(self, rewards_per_episode):
        # Save plots
        fig, ax1 = plt.subplots()

        # Plot average rewards per last 100 episodes , and the cumulative mean over all episodes (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])

        mean_total = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_total)):
            mean_total[x] = np.mean(rewards_per_episode[0:(x+1)])
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Mean Reward Last 100 Episodes', color='tab:blue')
        ax1.plot(mean_rewards, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create a second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative Mean Reward', color='tab:green')
        ax2.plot(mean_total, color='tab:green', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Make y axis 1 and 2 the same scale
        ax1.set_ylim([min(min(mean_rewards), min(mean_total)), max(max(mean_rewards), max(mean_total))])
        ax2.set_ylim(ax1.get_ylim())

        # Save the figure
        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--continue_training', help='Continue training mode', action='store_true')
    parser.add_argument('--render', help='Rendering mode', action='store_true')
    parser.add_argument('--use_gpu', help='Device mode', action='store_true')
    parser.add_argument('--endless', help='Endless mode', action='store_true')
    args = parser.parse_args()

    TD3 = Agent(args.train, args.endless, args.continue_training, args.render, args.use_gpu, hyperparameter_set=args.hyperparameters)
    TD3.run(args.train)