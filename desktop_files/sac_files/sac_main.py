#! /usr/bin/env python3

import numpy as np
import random
import gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import time
import argparse
import pandas as pd  # Import pandas for CSV saving
import drone_gym_gazebo_env_continuous
import rospy
from sac_agent import Agent
from sac_utils import plot_learning_curve
from datetime import timedelta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-env", type=str, default="droneEnv", help="Environment name")
    parser.add_argument("-info", type=str, help="Information or name of the run")
    parser.add_argument("-ep", type=int, default=2000, help="The amount of training episodes, default is 100")
    parser.add_argument("-actor_lr", type=float, default=3e-4, help="Learning rate of adapting the network weights")
    parser.add_argument("-critic_lr", type=float, default=3e-4, help="Learning rate of adapting the network weights")
    parser.add_argument("-fixed_alpha", type=float, help="entropy alpha value, if not chosen the value is learned by the agent")
    parser.add_argument("-hidden_size", type=int, default=256, help="Number of nodes per neural network hidden layer, default is 256")
    parser.add_argument("-print_every", type=int, default=100, help="Prints every x episodes the average reward over x episodes")
    parser.add_argument("-batch_size", type=int, default=256, help="Batch size, default is 256")
    parser.add_argument("-buffer_size", type=int, default=int(5e4), help="Size of the Replay memory, default is 1e6")
    parser.add_argument("-tau", type=float, default=0.005, help="Softupdate factor tau, default is 1e-2")
    parser.add_argument("-gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
    parser.add_argument("-max_t", type=int, default=500)
    parser.add_argument("-saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
    parser.add_argument('-root', type=str, default='/home/pm/catkin_ws/src/mavros-px4-vehicle/', help='root path for saving/loading')
    args = parser.parse_args()

    env_id = 'droneEnv'
    rospy.init_node('train_node', anonymous=True)
    env = gym.make('DroneGymGazeboEnvContinuous-v0')  # env = gym.make(env_id)
    
    filename = 'test' + str(args.ep) + '.png'
    figure_file = args.root + 'plots/' + filename
    save_load_dir = args.root + 'models/'

    # Initialize SAC agent
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    agent = Agent(state_size=(env.observation_space.shape), action_size=action_size, actor_lr=args.actor_lr,
                  critic_lr=args.critic_lr, gamma=args.gamma, fixed_alpha=args.fixed_alpha,
                  tau=args.tau, batch_size=args.batch_size, buffer_size=args.buffer_size,
                  hidden_size=args.hidden_size, action_prior="uniform")

    scores_deque = deque(maxlen=100)
    score_history = []

    # Lists to store data for CSV
    timesteps_array, episode_reward_array = [], []
    
    # Tracking times
    start_time = time.time()
    best_model_time = 0
    min_reward_time = None
    total_steps = 0
    best_avg_reward = -np.inf

    for i_episode in range(1, args.ep + 1):
        observation, _ = env.reset()
        score = 0
        for t in range(args.max_t):
            action = agent.act(observation)
            action_v = action.numpy()
            action_v = np.clip(action_v * action_high, action_low, action_high)
            observation_, reward, done, info = env.step(action_v)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn(t)
            score += reward
            observation = observation_
            total_steps += 1

            if done:
                break 
        
        scores_deque.append(score)
        score_history.append(np.mean(scores_deque))

        # Append episode data for CSV
        timesteps_array.append(total_steps)
        episode_reward_array.append(score)
        
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(scores_deque)

        # Check for reaching minimum reward if it hasn’t been recorded yet
        if min_reward_time is None and avg_reward >= 200:
            min_reward_time = time.time() - start_time  # Time to reach avg reward of 200
        
        # Check if we have a new best average reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward  # Update best average reward
            best_model_time = time.time() - start_time  # Record time to best model
        
        # Printing progress
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, avg_reward), end="")
        if i_episode % args.print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, avg_reward))
            
    # Save model
    torch.save(agent.actor_local.state_dict(), args.root + 'models/actor.pth')
    torch.save(agent.critic1.state_dict(), args.root + 'models/critic1.pth')
    torch.save(agent.critic2.state_dict(), args.root + 'models/critic2.pth')
    torch.save(agent.critic1_target.state_dict(), args.root + 'models/critic1_target.pth')
    torch.save(agent.critic2_target.state_dict(), args.root + 'models/critic2_target.pth')

    # Plot learning curve
    x = [i + 1 for i in range(args.ep)]
    plot_learning_curve(x, score_history, figure_file)

    # Save data to CSV
    data = {'timesteps': timesteps_array, 'episode_rewards': episode_reward_array}
    df = pd.DataFrame(data)
    df.to_csv(args.root + 'plots/SAC_data.csv', index=False)

    # Make the drone land
    env.land_disconnect_drone()
    rospy.logwarn("end")

    # Print time tracking results
    total_seconds = time.time() - start_time
    print("Total time taken: {:02}:{:02}:{:02}".format(int(total_seconds // 3600), int((total_seconds % 3600) // 60), int(total_seconds % 60)))
    print("Total time taken for best model: {}".format(timedelta(seconds=best_model_time)))
    if min_reward_time:
        print("Time taken to reach an minimum reward: {}".format(timedelta(seconds=min_reward_time)))
    else:
        print("Minimum reward was not achieved during training.")
    
    print("Training took {} min!".format((time.time() - start_time) / 60))
