import os
import pickle
from datetime import datetime
from time import time
from collections import deque, defaultdict
from joblib import Parallel, delayed
import numpy as np
import logging
import gym
import gym.spaces
import argparse
from utils import run_episode, calculate_returns
from agents import BaselineRandomPolicy, VanillaHillClimbingPolicy, SteepestAscendHillClimbingPolicy
from agents import SimulatedAnnealingPolicy, AdaptiveNoiseScalingPolicy

# create logger
logger = logging.getLogger('HillClimbingAgents')
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

filehandler = logging.FileHandler(filename='hillclimb.log')
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.DEBUG)
logger.addHandler(steamhandler)

def load_policy(policy_name, env, max_steps, gamma):
    if policy_name == 'random':
        logger.debug('loading baseline random policy')
        return BaselineRandomPolicy(env)
    elif policy_name == 'vanilla':
        logger.debug('loading vanilla hill climbing policy')
        return VanillaHillClimbingPolicy(env)
    elif policy_name == 'steepest':
        logger.debug('loading steepest ascend policy')
        return SteepestAscendHillClimbingPolicy(env, max_steps, gamma)
    elif policy_name == 'anneal':
        logger.debug('loading simulated annealing policy')
        return SimulatedAnnealingPolicy(env)
    elif policy_name == 'ada':
        logger.debug('loading adaptive noise scaling policy')
        return AdaptiveNoiseScalingPolicy(env)
    else:
        raise ValueError('No policy by name {} is available'.format(policy_name)) 

def record_train_output(args, policy, all_rewards, output_dir='runs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # save data
    output_data = {
        "args": args,
        "all_rewards": all_rewards
    }
    data_fname = '{}/data.pkl'.format(output_dir)
    pickle.dump(output_data, open(data_fname, 'wb'))
    logger.info('Output data saved to {}'.format(data_fname))

    # save policy
    policy_fname = '{}/policy.pkl'.format(output_dir)
    pickle.dump(policy, open(policy_fname, 'wb'))
    logger.info('Policy saved to {}'.format(data_fname))
    
def train(args, stop_on_solve=True, print_every=100):
    """
    Create gym environment and train a variant of hill climbing policy    
    """
    env = gym.make(args.env) 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # env_name, policy_name, num_episodes, max_steps, goal_score
    policy = load_policy(args.policy, env, args.steps, args.gamma)

    avg_return = deque(maxlen=100)
    all_rewards = []
    for episode in range(1, args.episodes+1):
        # rollout
        rewards = run_episode(policy, env, args.steps)
        # calculate discounted return
        current_return = calculate_returns(args.gamma, rewards)
        # update policy
        policy.step(current_return)
        # record returns
        avg_return.append(current_return)
        all_rewards.append(current_return)

        if episode % print_every == 0:
            logger.info('{}/{} average return of last 100 episodes {}'.format(episode, args.episodes, np.mean(avg_return)))

        if stop_on_solve:
            if np.mean(avg_return) >= args.goal:
                print('Env solved in {}'.format(episode))
                break

    return policy, all_rewards, np.mean(avg_return)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("env", help="Name of gym environment")    
    parser.add_argument("policy", type=str, help="Name of policy: [random, vanilla, steepest, anneal, ada]")
    parser.add_argument("episodes", type=int, help="Number of episodes to run")
    parser.add_argument("steps", type=int, help="Maximum number of timesteps to run in each episode")
    parser.add_argument("goal", type=int, help="Score when environment is considered solved")
    parser.add_argument("--env", type=str, default='CartPole-v0', help="Name of gym environment")
    parser.add_argument("--runs", type=int, default=1, help="Run the training for number of runs")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")

    
    args = parser.parse_args()    
    
    launched_at=datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    start = time()
    all_rewards = defaultdict()
    best_return = -np.inf
    best_policy = None
    
    # results = Parallel(n_jobs=10, verbose=10)(delayed(train)(filename) for i in range(args.runs))
    # embeddings.extend(results)        

    for i in range(args.runs):
        # train the policy
        policy, rewards, avg_return = train(args) 
        all_rewards[i] = rewards
        # we want to store only the best policy
        if avg_return >= best_return:
            best_policy = policy
            logger.info('Found best policy with average return {} at run {}'.format(avg_return, i))
        

    end = time()
    logger.info('Finished in {} seconds'.format(end - start))
    # record output and save policy
    output_dir = 'runs/{}/{}'.format(args.policy, launched_at)
    record_train_output(args, avg_return, all_rewards, output_dir=output_dir)

if __name__ == '__main__':
    main()