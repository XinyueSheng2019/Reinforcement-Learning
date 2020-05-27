from abc import ABC, abstractmethod
from copy import deepcopy
import gym
import numpy as np
import os.path
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam
from typing import Dict, Iterable, List


from rl2020.exercise4.networks import FCNetwork
from rl2020.exercise3.replay import Transition, ReplayBuffer


class Agent(ABC):

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space

        :attr saveables (Dict[str, torch.nn.Module]):
            mapping from network names to PyTorch network modules
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str) -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "{path}"

        :param path (str): path to directory where to save models
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        """Returns an action to select in given observation

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **DO NOT CHANGE THIS FUNCTION**

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        """Updates model parameters

        **DO NOT CHANGE THIS FUNCTION**
        """
        ...



class Actorcritic(Agent):
    """ The Actorcritic Agent for Ex 4

    **YOU MUST COMPLETE THIS CLASS**
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        gamma: float,
        **kwargs,
    ):
        """
        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param gamma (float): discount rate gamma

        :attr policy (FCNetwork): fully connected actor network for policy
        :attr policy_optim (torch.optim): PyTorch optimiser for policy network
        """
        super().__init__(action_space, observation_space)
        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n
       

        self.policy = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
        )

        self.policy_optim = Adam(self.policy.parameters(), lr=learning_rate, eps=1e-3)



        self.critic_net = FCNetwork(
             (STATE_SIZE, *hidden_size, 1), output_activation=None
        )

        self.critic_optim = Adam(self.critic_net.parameters(), lr=learning_rate, eps=1e-3)
        
        self.learning_rate = learning_rate
        self.gamma = gamma


        self.saveables.update({"policy": self.policy})

    def schedule_hyperparameters(self, timestep: int, max_timesteps: int):

        
        # max_deduct, decay = 0.95, 0.1
        # self.gamma = 1.0 - (min(1.0, timestep/(decay * max_timesteps))) * max_deduct
       

        

    def act(self, obs: np.ndarray, explore: bool):

        state = torch.from_numpy(obs).float().unsqueeze(0)
        probs = self.policy(state)
        probs = torch.nn.functional.softmax(probs)

        m = Categorical(probs)
        action = m.sample()
     
        
        return action.item(),m.log_prob(action)


    def update(
        self, rewards: List[float], states: List[np.ndarray], p_update: int, t: int, dones: List[bool], next_states: List[np.ndarray], log_probs:List[float],
    ) -> Dict[str, float]:
        """Update function for N_step Actor Critic
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
       
        if dones[t] == True:
            G = torch.zeros(1,1).type(torch.FloatTensor)
        else:
            next_obs = torch.from_numpy(next_states[t]).float()
            G = self.critic_net.forward(next_obs)
           
        actor_loss = 0.0
        value_loss = 0.0
      
        p = t
        while(p>=p_update):
            G = rewards[p] + self.gamma * G
            state = torch.from_numpy(states[p]).float().unsqueeze(0)
            value = self.critic_net(state)
            actor_loss = actor_loss - log_probs[p] * (G - value.detach())
            value_loss = value_loss + (G - value)**2
            p = p - 1

        actor_loss = actor_loss/len(rewards)
        value_loss = value_loss/len(rewards)

        self.policy_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.policy_optim.step()


        self.critic_optim.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()
       

        p_loss = 0
      

        return {"p_loss": p_loss}
