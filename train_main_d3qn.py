#! /usr/bin/env python3

from train_d3qn_agent import D3QNAgent
from utils import plot_learning_curve
import drone_gym_gazebo_env_discrete # defines 'DroneGymGazeboEnv-v0' custom environment

import gym
import numpy as np
import rospy
import argparse
import time
import pandas as pd

import torch

if __name__ == '__main__':
    # Allow the following arguments to be passed:
    parser = argparse.ArgumentParser(description='Drone StableBaselines')
    parser.add_argument('-load_checkpoints', type=bool, default=False)
    parser.add_argument('-render', type=bool, default=False)
    parser.add_argument('-eps_start', type=float, default=1, help='What epsilon starts at')    
    parser.add_argument('-eps_min', type=float, default=0.05, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-exploration_steps', type=float, default=75000, help='Number of steps before epsilon levels out at eps_min')
    parser.add_argument('-max_episodes', type=int, default=5000)
    parser.add_argument('-root_path', type=str, default='/home/pm/catkin_ws/src/mavros-px4-vehicle/', help='root path for saving/loading')   
    args = parser.parse_args()

    rospy.init_node('drone_node', anonymous=True)
    env = gym.make('DroneGymGazeboEnvDiscrete-v0')

    figure_file = args.root_path + 'plots/D3QN.png'
    save_load_dir = args.root_path + 'models/'

    eps_decay_rate = (args.eps_start - args.eps_min) / args.exploration_steps
    agent = D3QNAgent(n_actions=env.action_space.n,
                     input_dims=(env.observation_space.shape),
                     eps_start=args.eps_start, eps_min=args.eps_min, eps_dec=eps_decay_rate,
                     save_load_dir=save_load_dir)

    if args.load_checkpoints:
        agent.load_models()

    best_avg_reward = -np.inf
    avg_reward = 0
    total_steps = 0
    episode = 1
    episode_reward_array, eps_history_array, steps_array = [], [], []
    
    # Tracking time metrics
    start_time = time.time()
    min_reward_time = None

    while episode <= args.max_episodes:
        print(f'######################################################################')
        print(f'                           Episode = {episode}')
        print(f'######################################################################')

        done = False
        episode_reward = 0
        episode_steps = 0
        observation, _ = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            episode_reward += reward

            print('Episode ' + str(episode) + ', episode steps = ' + str(episode_steps) + ', total steps = ' + str(total_steps) + ', x_obs = ' + str(observation[-1, 0, 0]) + ', y_obs = ' + str(observation[-1, 0, 1]))
            
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_
            total_steps += 1
            episode_steps += 1
        
        episode_reward_array.append(episode_reward)
        steps_array.append(total_steps)
        eps_history_array.append(agent.epsilon)
        avg_reward = np.mean(episode_reward_array[-100:])
        
        print('Episode', episode, ' Summary: ep reward', episode_reward, ', average ep reward %.1f' % avg_reward, ', best avg reward %.2f' % best_avg_reward, ', epsilon %.2f' % agent.epsilon, ', ep steps ', episode_steps, ', tot steps ', total_steps)

        # Time tracking for reaching an average reward of 25
        if min_reward_time is None and avg_reward >= 25:
            min_reward_time = time.time() - start_time

        # Update best average reward and save model if applicable
        if avg_reward > best_avg_reward:
            agent.save_models()
            best_avg_reward = avg_reward

        episode += 1

    # Calculate total time
    total_seconds = time.time() - start_time
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Total time taken: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))

    # Plot learning curve
    x = [i + 1 for i in range(len(episode_reward_array))]
    plot_learning_curve(steps_array, episode_reward_array, eps_history_array, figure_file)

    # Save data to CSV
    data = {'timesteps': steps_array, 'episode_rewards': episode_reward_array, 'eps_history': eps_history_array}
    df = pd.DataFrame(data)
    df.to_csv(args.root_path + 'plots/D3QN_data.csv', index=False)

    # Print time tracking results
    print("Total training time: {:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds)))
    if min_reward_time:
        print("Time taken to reach an min reward (25): {}".format(time.strftime("%H:%M:%S", time.gmtime(min_reward_time))))
    else:
        print("Minimum reward was not achieved during training.")

    # Land the drone
    env.land_disconnect_drone()
    rospy.logwarn('finished :)')
