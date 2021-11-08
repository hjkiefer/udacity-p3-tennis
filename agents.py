import torch
torch.manual_seed(192736) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
from typing import List

class ddpg_agent():
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        
        self.actor_target = config["actor_network_function"]().to(self.device)
        self.actor_local = config["actor_network_function"]().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                          lr=config["learning_rate_actor"])
        self.soft_update(self.actor_local, self.actor_target, tau=1) # hard update to start networks on same values
        
        self.critic_target = config["critic_network_function"]().to(self.device)
        self.critic_local = config["critic_network_function"]().to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=config["learning_rate_critic"], 
                                           weight_decay=0)
        self.soft_update(self.critic_local, self.critic_target, tau=1) # hard update to start networks on same values
        
        self.memory = config["replay_buffer"]
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.action_selector = config["action_selector"]
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        for k in range(state.shape[0]):
            self.memory.add(state[k], action[k], reward[k], next_state[k], done[k])
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config["network_optimize_every_n_step"]
        if self.t_step == 0:
            for _ in range(self.config["learn_iterations"]):
                # If enough samples are available in memory, get random subset and learn
                experiences = self.memory.sample()
                if experiences:
                    #print("learning")
                    self.learn(experiences)
    
    def _convert_env_to_torch(self, variable):
        return torch.from_numpy(variable).float().to(self.device)
        
    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = self._convert_env_to_torch(state)
        self.actor_local.eval()  # set module in evaluation mode
        with torch.no_grad():
            action = self.actor_local(state)
        self.actor_local.train()  # set module in training mode

        # Select action
        return self.action_selector.select_action(action)
    
    def learn(self, experiences):
        gamma = self.config["gamma_discount_factor"]
        solved_score = self.config["environment_solved_score"]
        critic_max_grad_norm = 1.
        actor_max_grad_norm = 1.
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.critic_target(next_states, self.actor_target(next_states))
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        
        value_loss = F.mse_loss(Q_expected, Q_targets)
        # perform update of both actor and critic using this loss
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), critic_max_grad_norm)
        self.critic_optimizer.step()
                
        self.actor_optimizer.zero_grad()
        a = self.actor_local(states)
        policy_loss = - self.critic_local(states, a).mean()
        policy_loss.backward()

        self.actor_optimizer.step()

        # ------------------- update target networks ------------------- #
        self.soft_update(self.actor_local, self.actor_target)     
        self.soft_update(self.critic_local, self.critic_target)
        
    def soft_update(self, local_model, target_model, tau=None):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        if not tau:
            tau = self.config["tau_soft_update"]
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
