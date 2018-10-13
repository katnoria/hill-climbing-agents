from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from utils import run_episode, calculate_returns

class Policy(ABC):
    """Abstract Base Class to define Policy"""
    def __init__(self, env, seed=0):
        np.random.seed(seed)
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.w = 1e-4 * np.random.rand(self.state_size, self.action_size)
        self.best_return = -np.inf
        self.best_weights = self.w
        
    def forward(self, state):
        """
        Forward pass to calculate the action probabilities

        Parameters
        ----------
        state (array_like): gym env state
        """
        x = np.dot(state, self.w)
        # softmax
        x = np.exp(x)/sum(np.exp(x))
        return x

    def act(self, state):
        """
        Get the action of take on specified state

        Parameters
        ----------
        state (array_like): gym env state
        """
        probs = self.forward(state)
        action = np.argmax(probs)
        return action

    @abstractmethod
    def step(self, current_return):
        """Update policy according to the update schedule/logic"""
        pass


class VanillaHillClimbingPolicy(Policy):
    """Implementation of vanilla hill climbing"""
    def __init__(self, env, noise=1e-4, seed=0):
        super(VanillaHillClimbingPolicy, self).__init__(env, seed)
        self.noise = noise

    def step(self, current_return):
        """
        Perturb policy weights by predefined radius (noise) and
        update best return and weights if found

        Parameters
        ----------
        current_return (int): Return of current rollout
        """
        super().step(current_return)
        if current_return >= self.best_return:
            self.best_return = current_return
            self.best_weights = self.w
        # update weights
        self.w = self.best_weights + self.noise * np.random.rand(*self.best_weights.shape)


class SteepestAscendHillClimbingPolicy(Policy):
    """Implementation of Steepest ascend hill climbing"""
    def __init__(self, env, max_steps, gamma, neighbors=4, noise=1e-4, seed=0):
        super(SteepestAscendHillClimbingPolicy, self).__init__(env, seed)        
        self.neighbors = neighbors
        self.gamma = gamma 
        self.max_steps = max_steps
        self.noise = noise

    def step(self, current_return):
        """
        Perturb policy weights by generating neighboring policies and
        comparing the return from each policy with current best.
        if the return from any of neighboring policy is greater than 
        or equal to current best then we use that policy as our current.

        Parameters
        ----------
        current_return (int): Return of current rollout
        """        
        super().step(current_return)
        # Check the return from all neighbors
        candidate_returns = [current_return]
        candidate_weights = [self.w]        
        for _ in range(self.neighbors):
            policy = deepcopy(self)
            policy.w = self.best_weights + self.noise * np.random.rand(*self.best_weights.shape)
            rewards = run_episode(policy, self.env, self.max_steps)
            policy_return = calculate_returns(self.gamma, rewards)
            candidate_returns.append(policy_return)
            candidate_weights.append(policy.w)

        # Find the max return from candidate returns and 
        # compare it with our best return
        best_idx = np.argmax(np.array(candidate_returns))
        if candidate_returns[best_idx] >= self.best_return:
            self.best_return = candidate_returns[best_idx]
            self.best_weights = candidate_weights[best_idx]
            self.w = candidate_weights[best_idx]


class SimulatedAnnealingPolicy(Policy):
    """Implementation of simulated annealing"""
    def __init__(self, env, noise=1e-4, min_noise=1e-4, seed=0):
        """Initialise poicy

        Parameters
        ----------
        env: gym environment
        noise: controls the radius of weights perturbation
        min_noise: the lower limit of how much do we allow radius to shrink
        seed: random seed
        """
        super(SimulatedAnnealingPolicy, self).__init__(env, seed)
        self.noise = noise
        self.min_noise = min_noise

    def step(self, current_return):
        """Anneal the noise per schedule"""
        super().step(current_return)
        if current_return >= self.best_return:
            self.best_return = current_return
            self.best_weights = self.w
            # schedule to anneal the noise
            self.noise = max(self.min_noise, self.noise/2)
        # update weights
        self.w = self.best_weights + self.noise * np.random.rand(*self.best_weights.shape)
        

class AdaptiveNoiseScalingPolicy(Policy):
    """Implementation of Adaptive noise scaling"""
    def __init__(self, env, noise=1e-2, max_noise=2, min_noise=1e-4, seed=0):
        """Initialise policy"""
        super(AdaptiveNoiseScalingPolicy, self).__init__(env, seed)
        self.noise = noise
        self.max_noise = max_noise
        self.min_noise = min_noise

    def step(self, current_return):
        super().step(current_return)
        if current_return > self.best_return:
            self.best_return = current_return
            self.best_weights = self.w
            # contract radius, if we are doing well
            self.noise = max(self.min_noise, self.noise/2)
        else:
            # expand radius, if we are not doing well
            self.noise = min(self.max_noise, self.noise * 2)
        # update weights
        self.w = self.best_weights + self.noise * np.random.rand(*self.best_weights.shape)    
