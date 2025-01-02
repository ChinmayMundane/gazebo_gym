"defines the policy for the PPO agent"
"A class for network structure and forward propogation is defined"
"The PPO agent is defined which creates actor, selects an action by running inputs through the actor network, and trains it"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# build the actor and critic deep networks
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor_mean = nn.Linear(128, output_dim)
        self.actor_log_std = nn.Linear(128, output_dim)  # Log std for numerical stability
        self.critic = nn.Linear(128, 1)

    def forward(self, x): # x is state. Cannot input any inf or nan values
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_mean = self.actor_mean(x) # run the network through the final actor layer
        action_log_std = self.actor_log_std(x)
        action_std = torch.exp(action_log_std)
        state_value = self.critic(x) # run the network through the final critic layer to get the value function
        return action_mean, action_std, state_value

class PPOAgent:
    def __init__(self, env, learning_rate=1e-4, gamma=0.99, epsilon=0.2, k_epochs=10): # adjust training hyperparameters here
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.k_epochs = k_epochs
        
        self.actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]) # Initialize the actor-critic network with the appropriate dimensions
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate) # Initialize optimizer to update the gradients in train(). Adam (Adaptive Moment Estimation) is like stochastic gradient descent but...
        
        # Another network to store the old policy to compare to the updated policy
        self.policy_old = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
        self.policy_old.load_state_dict(self.actor_critic.state_dict()) 
    
    # Select action of robot based on the current policy and state (sensor readings). Initializes as random, but learns to make correct decisions in training portion
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # convert state to tensor and add batch dimension () so we can run it through the network
        with torch.no_grad(): # disable gradient calculation to just inference: no backpropogation because we don't want to update here
            action_mean, action_std, _ = self.policy_old(state)
        action_dist = torch.distributions.Normal(action_mean, action_std) # create a normal distribution based on this mean and std 
        action = action_dist.sample() # sample from the gaussian
        action_log_prob = action_dist.log_prob(action) 
        return action.numpy(), action_log_prob.numpy()


    def discount_rewards(self, rewards, dones):
        discounted_rewards = torch.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                G = 0
            G = rewards[t] + self.gamma * G
            discounted_rewards[t] = G
        return discounted_rewards


    
    def train(self, memory, timesteps):
        states, actions, rewards, dones, log_probs_old = zip(*memory) # extract data from the buffer
        
        # Convert lists to numpy arrays
        states = np.array(states)
        actions = np.array(actions).reshape(timesteps, 2)
        rewards = np.array(rewards)
        dones = np.array(dones)
        log_probs_old = np.array(log_probs_old).reshape(timesteps, 2)

        # Convert numpy arrays to tensors
        states = torch.tensor(states, dtype=torch.float32)  # size: max_timesteps x num_states
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).sum(axis=1)
        
        discounted_rewards = self.discount_rewards(rewards, dones)
        
        for _ in range(self.k_epochs):
            action_means, action_stds, state_values = self.actor_critic(states) # get output of actor and critic network
            action_dists = Normal(action_means, action_stds)
            log_probs = action_dists.log_prob(actions).sum(axis=1)
            
            ratios = torch.exp(log_probs - log_probs_old) # probability ratio for old and new policies. How much does the new policy deviate from the old policy.
            advantages = discounted_rewards - state_values.squeeze() # How much better it is to take the specific action compared to the average actions (this is the critic critiquing the actor)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages # clipped surrogate loss to prevent instability
            
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (discounted_rewards - state_values.squeeze()).pow(2).mean() # loss = policy loss + value loss
            
            self.optimizer.zero_grad() # clear gradients after each epoch
            loss.backward() # compute gradients (back propogation)
            self.optimizer.step() # update the weights of the NN
        
        self.policy_old.load_state_dict(self.actor_critic.state_dict()) # update old policy as the new one
