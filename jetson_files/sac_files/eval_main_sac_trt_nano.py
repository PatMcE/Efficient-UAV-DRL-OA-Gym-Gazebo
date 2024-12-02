#! /usr/bin/env python3

import drone_gym_gazebo_env_continuous  # Custom environment file
import gym
import numpy as np
import rospy
import argparse
import time
import tensorrt as trt
from trt_utils import allocate_buffers, do_inference
import psutil
from jtop import jtop

# Set up TensorRT Logger and model path
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = '/home/jetson/catkin_ws/src/mavros-px4-vehicle/models/sac_half.trt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drone SAC Evaluation')
    parser.add_argument('-episodes', type=int, default=1)#100 episodes used to determine inference latency/success rate
    args = parser.parse_args()

    # Initialize ROS node and custom gym environment
    rospy.init_node('drone_node', anonymous=True)
    env = gym.make('DroneGymGazeboEnvContinuous-v0')

    # Load TensorRT engine for SAC Actor Network
    with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)  # Allocate TensorRT buffers
    
    # Tracking performance metrics
    total_steps = 0
    successful_steps = 0
    num_exceeded_workspace = 0
    num_collided = 0
    num_reached_des_point = 0
    accumulated_inf_time = 0
    
    with jtop() as jetson:        
        for episode in range(1, args.episodes + 1):
            print('######################################################################')
            print(f'                           Episode = {episode}')
            print('######################################################################')

            # Reset environment and episode variables
            done = False
            episode_reward = 0
            episode_steps = 0
            observation, _ = env.reset()
            
            while not done:
                start_time = time.time()
                
                # Preprocess observation for TensorRT
                preprocessed_obs = np.ascontiguousarray(observation.reshape((1,) + observation.shape))
                np.copyto(inputs[0].host, preprocessed_obs.ravel())  # Copy observation to input buffer

                # Run inference on the TensorRT engine
                trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                
                # Extract `mu` and `log_std` from TensorRT outputs
                mu = trt_outputs[0]
                log_std = np.clip(trt_outputs[1], -20, 2)
                std = np.exp(log_std)
                
                # Sample action using the reparameterization trick (SAC policy)
                epsilon = np.random.normal(size=mu.shape).astype(np.float32)
                action = 0.5 * np.tanh(mu + std * epsilon)  # Apply tanh for bounded action

                # Measure inference time
                accumulated_inf_time += time.time() - start_time

                # Step environment with the sampled action
                observation_, reward, done, info = env.step(action)
                
                # Log episode statistics
                episode_reward += reward
                observation = observation_
                total_steps += 1
                episode_steps += 1
                
                print(f'Episode {episode}, Step {episode_steps}, Reward: {reward}, Total Steps: {total_steps}')
                #time.sleep(0.3)
                #rate.sleep()

            # Update metrics after the episode ends
            if env.get_has_drone_exceeded_workspace():
                num_exceeded_workspace += 1
            if env.get_has_drone_collided():
                num_collided += 1
            if env.get_has_reached_des_point():
                num_reached_des_point += 1
                successful_steps += episode_steps
            
            print(f'Episode {episode} Summary: Steps {episode_steps}, Reward {episode_reward}, Total Steps {total_steps}')
            print(f'Exceeded Workspace: {num_exceeded_workspace}, Collided: {num_collided}, Reached Destination: {num_reached_des_point}')

    # Display overall performance after all episodes
    print(f'Success Rate: {(num_reached_des_point / args.episodes) * 100:.2f}%')
    print(f'Average Inference Time: {(accumulated_inf_time / total_steps) * 1000:.2f} ms')
    print(f'Average Steps in Successful Episodes: {successful_steps / max(1, num_reached_des_point)}')

    # Land the drone safely
    env.land_disconnect_drone()
    rospy.logwarn('Evaluation Complete :)')
