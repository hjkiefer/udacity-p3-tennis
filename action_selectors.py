import numpy as np
import torch

####################
### Noise function

class NoisyActionSelector():
    def __init__(self,
                 decay_rate=0.995,
                 init_noise_scale=1,
                 auto_update_scale=False,
                 action_bounds=None,
                 device="cpu",
                 minimum_noise=None):
        self.decay_rate=decay_rate
        self.noise_scale=init_noise_scale
        self.action_bounds = action_bounds
        self.auto_update_scale = auto_update_scale
        self.device = device
        self.minimum_noise=minimum_noise
        
    def select_action(self, model_greedy_action):
        noise = np.random.normal(loc=0,
                                 scale=self.noise_scale,
                                 size=model_greedy_action.shape)
        noisy_action = model_greedy_action + torch.Tensor(noise).to(self.device)
        if self.action_bounds:
            noisy_action = np.clip(noisy_action, self.action_bounds[0], self.action_bounds[1])
        if self.auto_update_scale:
            self.noise_scale_update()
        return noisy_action
    
    def noise_scale_update(self):
        self.noise_scale = self.noise_scale*self.decay_rate
        if self.minimum_noise:
            self.noise_scale = max([self.minimum_noise,self.noise_scale])
    
    def step_next_episode(self):
        self.noise_scale_update()