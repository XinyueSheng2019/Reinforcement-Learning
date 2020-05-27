import gym
import numpy as np
import time
from tqdm import tqdm

from rl2020.exercise4.agents import Actorcritic


LUNARLANDER_CONFIG_1 = {
    "env": "LunarLander-v2",
    "episode_length": 250,
    "target_return": 190.0,
    "max_timesteps": 1e6,
    "eval_freq": 50000,
    "eval_episodes": 10,
    "max_time": 180 * 60,
    "gamma": 0.99,
    "hidden_size": (64, 64),
    "learning_rate": 0.0003,
    "lookahead_steps":1,

    "save_filename": None,
}
LUNARLANDER_CONFIG_10 = {
    "env": "LunarLander-v2",
    "episode_length": 250,
    "target_return": 190.0,
    "max_timesteps": 1e6,
    "eval_freq": 50000,
    "eval_episodes": 10,
    "max_time": 180 * 60,
    "gamma": 0.99,
    "hidden_size": (64, 64),
    "learning_rate": 0.0003,
    "lookahead_steps":10,

    "save_filename": None,
}

CARTPOLE_CONFIG_1 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.99,
    "hidden_size": (64,),
    "learning_rate": 0.004,
    "lookahead_steps":1,
    "save_filename": None,
}

CARTPOLE_CONFIG_10 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.95,
    "hidden_size": (64,),
    "learning_rate": 0.006,
    "lookahead_steps":10,
    "save_filename": None,
}

CARTPOLE_CONFIG_50 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.95,
    "hidden_size": (64,),
    "learning_rate": 0.006,
    "lookahead_steps":50,
    "save_filename": None,
}

CARTPOLE_CONFIG_100 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.95,
    "hidden_size": (64,),
    "learning_rate": 0.006,
    "lookahead_steps":100,
    "save_filename": None,
}

CARTPOLE_CONFIG_150 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.95,
    "hidden_size": (64,),
    "learning_rate": 0.006,
    "lookahead_steps":150,
    "save_filename": None,
}

CARTPOLE_CONFIG_200 = {
    "env": "CartPole-v1",
    "episode_length": 200,
    "target_return": 195.0,
    "max_timesteps": 200000,
    "eval_freq": 2000,
    "eval_episodes": 20,
    "max_time": 30 * 60,
    "gamma": 0.95,
    "hidden_size": (64,),
    "learning_rate": 0.006,
    "lookahead_steps":200,
    "save_filename": None,
}


# CONFIG = CARTPOLE_CONFIG_50
CONFIG = LUNARLANDER_CONFIG_1

RENDER = False


def play_episode(
    env: gym.Env,
    agent: Actorcritic,
    N: int,
    train: bool = True,
    explore=True,
    render=False,
    max_steps=200,
    
):
    """
    Play out a single episode with REINFORCE

    :param env (gym.Env): gym environment to use
    :param agent (Reinforce): REINFORCE agent to train
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether agent should use exploration
    :param render (bool): flag whether environment steps should be rendered
    :param max_steps (int): maximum number of steps to take for the episode
    :return (Tuple[int, float]): number of timesteps completed during episode, episode return
    """
    obs = env.reset()

    if render:
        env.render()

    done = False
    num_steps = 0
    total_rewards = 0

    states = []
    rewards = []
    dones = []
    next_states = []
    log_probs = []
  
    # p_update = 0

    while not done and num_steps < max_steps:
        action, log_prob= agent.act(np.array(obs), explore=explore)
        nobs, rew, done, _ = env.step(action)

        states.append(obs)
        rewards.append(rew)
        dones.append(done)
        next_states.append(nobs)
        log_probs.append(log_prob)

        if render:
            env.render()
            
        num_steps += 1
        total_rewards += rew

        obs = nobs
    
    if train:
        p_update = 0
        for step in range(len(rewards)):
            if step - p_update == N or dones[step] == True:
                loss = agent.update(rewards, states, p_update, step, dones, next_states, log_probs)
                p_update = step


    
    return num_steps, total_rewards


def train(env, config, output=True):
    """
    Execute training of N-step Actor Critic on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): mean average returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = Actorcritic(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
   
    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)

            num_steps, _ = play_episode(
                env,
                agent,
                N=config["lookahead_steps"],
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                
            )


            timesteps_elapsed += num_steps

            pbar.update(num_steps)

            #evaluate the policy
            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                for _ in range(config["eval_episodes"]):
                    _, total_reward = play_episode(
                        env,
                        agent,
                        N=config["lookahead_steps"],
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=env._max_episode_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])
                print(eval_return)
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_times_all.append(time.time() - start_time)
                if eval_return >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_return} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    returns, times = train(env, CONFIG)
    env.close()
