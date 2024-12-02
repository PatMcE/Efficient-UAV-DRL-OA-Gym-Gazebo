'''
This code uses code from from https://github.com/edowson/openai_ros and https://github.com/ashdtu/openai_drone_gym
'''

import gym
from gym.utils import seeding
from gym import spaces
from gym.envs.registration import register

import numpy as np
import math
import time
import rospy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion, Twist, Vector3
from std_srvs.srv import Empty

from cv_bridge import CvBridge, CvBridgeError
import cv2

from mavros_px4_vehicle.px4_modes import PX4_MODE_OFFBOARD
from mavros_px4_vehicle.px4_offboard_modes import SetPositionWithYawCmdBuilder
from mavros_px4_vehicle.px4_vehicle import PX4Vehicle

from gazebo_msgs.srv import (
    SetModelState,
    GetModelState,
    GetModelProperties
)

from gazebo_msgs.msg import ModelState

# Register the custom gym environment
timestep_limit_per_episode = 225
register(
    id='DroneGymGazeboEnvDiscrete-v0',
    entry_point='drone_gym_gazebo_env_discrete:DroneGymGazeboEnvDiscrete',
    max_episode_steps=timestep_limit_per_episode,
)

class DroneGymGazeboEnvDiscrete(gym.Env):
	def __init__(self, drone_name="drone1", render_mode="human"):
		self.drone_name = drone_name
		self.drone = PX4Vehicle(name=self.drone_name, auto_connect=True) #drone treated as an object thanks to 'mavros_px4_vehicle'
		self.z = 1.0 #height the drone flies at

		self.action_duration = 0.5 #length of time of moving forward/left/right in a single step (in seconds)        
		self.action_space = spaces.Discrete(3) #3 actions = forward/left/right
		self.end_episode_reward = 100
		self.height = 80 #height want image to be after processing (original image = 480)
		self.width = 106 #width want image to be after processing (original image = 640)
		self.one_image_shape = (1, self.height, self.width) #(channels, height, width)
		self.three_image_shape = (3, self.height, self.width) #(channels, height, width)
		self.observation_space = spaces.Box(low=-1.5, high=6.0, shape=self.three_image_shape, dtype=np.float32)

		# Set starting and goal/desired point:
		self.start_point = Point()
		self.start_point.x = 0.0
		self.start_point.y = 0.0
		self.start_point.z = self.z
		self.desired_point = Point()
		self.desired_point.x = 5.0
		self.desired_point.y = self.start_point.y
		self.desired_point.z = self.start_point.z

		# Set workspace limits:
		self.work_space_x_max = 5.35
		self.work_space_x_min = -1.0
		self.work_space_y_max = 2.25
		self.work_space_y_min = -2.25
		self.work_space_z_max = 3.2
		self.work_space_z_min = -0.1

		# Additional setup:
		self.bridge = CvBridge()
		self.episode_num = 0
		self.cumulated_episode_reward = 0
		self.speed = 0.5 #drone speed (m/s)
		self.y_start_point_range = [-1.0, 1.0]
		self.possible_x_start_points  = [0.0, -0.3]
		self.possible_y_start_desired_points  = [-1.5, -0.75, 0.0, 0.75, 1.5] #used to randomize y starting and y desired position of drone everytime episode reset (see reset() function)
		rospy.Subscriber(f"/{self.drone_name}/camera/depth/image_raw", Image, self._front_camera_depth_image_raw_callback)                
		rospy.Subscriber(f"/{self.drone_name}/mavros/local_position/pose", PoseStamped, self._gt_pose_callback)
		self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self._check_front_camera_depth_image_raw_ready()
		self._check_gt_pose_ready()
		self._init_env_variables()

	def unpause_sim(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		try:
			self.unpause()
		except rospy.ServiceException as e:
			print ("/gazebo/unpause_physics service call failed")

	def pause_sim(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		try:
			self.pause()
		except rospy.ServiceException as e:
			print ("/gazebo/pause_physics service call failed")

	def reset_dist_to_desired_point_and_observations(self):
		self.previous_distance_from_des_point = self.get_distance_from_desired_point(self.get_gt_pose().pose.position)

		self.t_minus1_img_obs = np.zeros(self.one_image_shape, dtype=np.float32)
		self.position_img_obs = np.zeros(self.one_image_shape, dtype=np.float32)
		self.obs = self._get_obs()
		
	def _init_env_variables(self):
		self.cumulated_steps = 0.0
		self.cumulated_reward = 0.0

		self.reset_dist_to_desired_point_and_observations()

	def _check_front_camera_depth_image_raw_ready(self):
		self.front_camera_depth_image_raw = None
		rospy.logdebug(f"Waiting for /"+str(self.drone_name)+"/camera/depth/image_raw to be READY...")
		while self.front_camera_depth_image_raw is None and not rospy.is_shutdown():
			try:
				self.front_camera_depth_image_raw = rospy.wait_for_message("/" + str(self.drone_name) + "/camera/depth/image_raw", Image, timeout=5.0)
				rospy.logdebug("Current /"+str(self.drone_name)+"/camera/depth/image_raw READY=>")
			except:
				rospy.logerr("Current "+str(self.drone_name)+"/camera/depth/image_raw not ready yet, retrying for getting front_camera_depth_image_raw")
		return self.front_camera_depth_image_raw

	def _check_gt_pose_ready(self):
		self.gt_pose = None
		rospy.logdebug("Waiting for /"+str(self.drone_name)+"/mavros/local_position/pose to be READY...")
		while self.gt_pose is None and not rospy.is_shutdown():
			try:
				self.gt_pose = rospy.wait_for_message("/"+str(self.drone_name)+"/mavros/local_position/pose", PoseStamped, timeout=5.0)
				rospy.logdebug("Current /"+str(self.drone_name)+"/mavros/local_position/pose READY=>")
			except:
				rospy.logerr("Current /"+str(self.drone_name)+"/mavros/local_position/pose not ready yet, retrying for getting gt_pose")
		return self.gt_pose

	def _front_camera_depth_image_raw_callback(self, image):
		# ROS Callback function for the /drone_name/camera/depth/image_raw topic
		self.front_camera_depth_image_raw = image

	def _gt_pose_callback(self, data):
		# ROS Callback function for the /drone_name/mavros/local_position/pose topic
		self.gt_pose = data

	def get_front_camera_depth_image_raw(self):
		return self.front_camera_depth_image_raw

	def get_gt_pose(self):
		return self.gt_pose

	def forward(self):
		gt_pose = self.get_gt_pose()        
		x = round(gt_pose.pose.position.x, 2)
		y = round(gt_pose.pose.position.y, 2)

		x_cmd = SetPositionWithYawCmdBuilder.build(x = x + round((self.speed),2), y = y, z = self.start_point.z)
		self.drone.set_pose2d(x_cmd)

		return time.time()

	def left_or_right(self, direction):#dir = 1 for left
		gt_pose = self.get_gt_pose()        
		x = round(gt_pose.pose.position.x, 2)
		y = round(gt_pose.pose.position.y, 2)

		y_cmd = SetPositionWithYawCmdBuilder.build(x = x, y = y + round((direction*self.speed),2), z = self.start_point.z)
		self.drone.set_pose2d(y_cmd)

		return time.time()

	def _set_action(self, action):
		if action == 0:
			#rospy.loginfo("MOVING FORWARDS")
			# Move forward at self.speed m/s for self.action_duration seconds
			start = self.forward()
			while self.action_duration > time.time() - start:
				pass

		if action == 1:
			#rospy.loginfo("MOVING LEFT")
			# Move left at self.speed m/s for self.action_duration seconds
			start = self.left_or_right(1)#1 indicates move left (not right)
			while self.action_duration > time.time() - start:
				pass

		if action == 2:
			#rospy.loginfo("MOVING RIGHT")
			# Move right at self.speed m/s for self.action_duration seconds
			start = self.left_or_right(-1)#-1 indicates move right (not left)
			while self.action_duration > time.time() - start:
				pass

	def depth_imgmsg_to_cv2(self, img_msg):
		try:
			depth_im = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
		except CvBridgeError as e:
			print(e)
		return depth_im

	def preprocess_image(self, image):
		#Convert to opencv format:
		cv_image = self.depth_imgmsg_to_cv2(image)#480,640

		# Normalize, resize, reshape and transpose:
		img_normalized = cv2.normalize(cv_image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		img_resized = cv2.resize(img_normalized, (self.one_image_shape[2], self.one_image_shape[1]), interpolation = cv2.INTER_CUBIC)
		img_reshaped = img_resized.reshape((self.one_image_shape[1], self.one_image_shape[2], self.one_image_shape[0]))
		img_transposed = img_reshaped.transpose(2, 0, 1)#transpose to the form (channel,height,width) for pytorch

		return_img = np.clip(img_transposed, 0, 1)

		return return_img #can get small negative values (e.g -0.1 if do not do this)

	def _get_obs(self):
		gt_pose = self.get_gt_pose()
		self.position_img_obs[0, 0, 0] = np.float32(round(self.desired_point.x - gt_pose.pose.position.x, 2))
		self.position_img_obs[0, 0, 1] = np.float32(round(self.desired_point.y - gt_pose.pose.position.y, 2))

		ROS_img_msg = self.get_front_camera_depth_image_raw()
		img_obs = self.preprocess_image(ROS_img_msg)

		combined_img_obs = np.concatenate((img_obs, self.t_minus1_img_obs, self.position_img_obs), axis=0)
		self.t_minus1_img_obs = img_obs

		return combined_img_obs

	def is_in_desired_position(self, current_position, epsilon=0.5):
		"""
		Return True if the current position is near desired poistion
		"""

		is_in_desired_pos = False

		x_pos_plus = self.desired_point.x + epsilon
		x_pos_minus = self.desired_point.x - epsilon
		y_pos_plus = self.desired_point.y + epsilon
		y_pos_minus = self.desired_point.y - epsilon

		x_current = current_position.x
		y_current = current_position.y

		x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
		y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

		is_in_desired_pos = x_pos_are_close and y_pos_are_close

		return is_in_desired_pos

	def get_distance_from_desired_point(self, current_position):
		a = np.array((current_position.x, current_position.y, current_position.z))
		b = np.array((self.desired_point.x, self.desired_point.y, current_position.z))#not self.desired_point.z as no need to consider z difference

		distance = np.linalg.norm(a - b)

		return distance

	def is_inside_workspace(self, current_position):
		is_inside = False

		if current_position.x > self.work_space_x_min and current_position.x <= (self.work_space_x_max+self.start_point.x):
			if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
				#if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
					is_inside = True

		if not(is_inside):
			rospy.logwarn("drone has exceeded workspace bounds")

		return is_inside

	def drone_has_collided(self, roll, pitch, yaw):
		"""
		When the drone moves left/right/forward/up/down its roll, pitch and yaw remain at approximatley 0.
		If the roll/pitch/yaw deviates much from 0 we know it has hit an object
		"""
		has_collided = True

		self.max_roll = 0.25
		self.max_pitch = 0.25
		self.max_yaw = 0.035

		if roll > -1*self.max_roll and roll <= self.max_roll:
			if pitch > -1*self.max_pitch and pitch <= self.max_pitch:
				if yaw > -1*self.max_yaw and yaw <= self.max_yaw:
					has_collided = False
		if (has_collided):
			rospy.logwarn("drone has collided. roll, pitch, yaw = " + str(round(roll,2)) + ", " + str(round(pitch,2)) + ", " + str(round(yaw,2)))
		else:
			rospy.logwarn("not collided, yaw = " + str(round(yaw,2)))

		return has_collided

	def _is_done(self, observations):
		current_position = Point()
		current_position.x = round(self.desired_point.x - observations[-1, 0, 0], 2) #observation[1][0] = x position observation
		current_position.y = round(self.desired_point.y - observations[-1, 0, 1], 2) #observation[1][1] = y position observation
		current_position.z = self.desired_point.z

		gt_pose = self.get_gt_pose()
		roll, pitch, yaw = self.euler_from_quaternion(gt_pose.pose.orientation.x, gt_pose.pose.orientation.y, gt_pose.pose.orientation.z, gt_pose.pose.orientation.w)

		is_inside_workspace_now = self.is_inside_workspace(current_position)
		has_drone_collided = self.drone_has_collided(roll, pitch, yaw)
		has_reached_des_point = self.is_in_desired_position(current_position)#default epsilon = 0.5

		episode_done = not(is_inside_workspace_now) or has_drone_collided or has_reached_des_point

		return episode_done	

	def _compute_reward(self, observations, episode_done):
		current_position = Point()
		current_position.x = round(self.desired_point.x - observations[-1, 0, 0], 2) #observation[1][0] = x position observation
		current_position.y = round(self.desired_point.y - observations[-1, 0, 1], 2) #observation[1][1] = y position observation
		current_position.z = self.desired_point.z

		distance_from_des_point = self.get_distance_from_desired_point(current_position)

		if not episode_done:
			reward = -1 + (self.previous_distance_from_des_point - distance_from_des_point)
		#if not episode_done:
			#if (distance_from_des_point < self.previous_distance_from_des_point):#i.e getting closer to dest point
				#reward = 1 + 1*(self.previous_distance_from_des_point - distance_from_des_point)
			#else:
				#reward = -1 + 1*(self.previous_distance_from_des_point - distance_from_des_point)
			#print('episode not done reward = ' + str(reward))
		else:
			if self.is_in_desired_position(current_position):
				reward = self.end_episode_reward
				rospy.logwarn("##############")
				rospy.logwarn("in desired pos")
				rospy.logwarn("##############")
			else:
				reward = (-1*self.end_episode_reward)# + (-20*distance_from_des_point)#self.end_episode_reward
				print('not desired position reward = ' + str(reward))

		self.previous_distance_from_des_point = distance_from_des_point
		self.cumulated_reward += reward
		self.cumulated_steps += 1

		return reward

	def step(self, action):
		self._set_action(action)

		self.obs = self._get_obs()
		done = self._is_done(self.obs)
		reward = self._compute_reward(self.obs, done)

		return self.obs, reward, done, {}

	def render(self, mode="human"):
		# Process the image to display
		ROS_img_msg = self.get_front_camera_depth_image_raw()
		img_obs = self.preprocess_image(ROS_img_msg)

		# Convert to a format suitable for displaying with OpenCV (cv2.imshow)
		img_display = img_obs[0]  # Assuming img_obs is in shape (1, height, width)
		img_display = (img_display * 255).astype(np.uint8)  # Rescale to 0-255 for display

		# Resize for better display
		img_display_resized = cv2.resize(img_display, (self.width*10, self.height*10), interpolation=cv2.INTER_NEAREST)

		# Display the image
		cv2.imshow("Depth Image", img_display_resized)#img_display_resized)
		cv2.waitKey(1)  # Add a short delay to allow the window to be updated

	# Reset model states, i.e. position and velocity of the ifo001 model
	def set_state_gazebo_model(self, model, pose, twist=Twist(linear=Vector3(x=0.0,y=0.0,z=0.0), angular=Vector3(x=0.0,y=0.0,z=0.0))):
		try:
			rospy.wait_for_service('/gazebo/set_model_state')
			set_model = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			state = ModelState()
			state.model_name = model
			state.pose = pose
			state.twist = twist
			state.reference_frame = "world"
			resp_set_model=set_model(state)
			#rospy.loginfo(f'Set model state: {resp_set_model}')
		except rospy.ServiceException as e:
			rospy.loginfo(f"Set Model State service call failed: {e}")

	def disable_collisions(self, model_name):
		rospy.wait_for_service('/gazebo/get_model_state')
		try:
			get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
			model_state = get_model_state(model_name, 'world')

			# Modify the model_state to have zero velocity and slightly raise it
			model_state.pose.position.z += 0.5  # Adjust this value as necessary
			model_state.twist.linear.x = 0
			model_state.twist.linear.y = 0
			model_state.twist.linear.z = 0
			model_state.twist.angular.x = 0
			model_state.twist.angular.y = 0
			model_state.twist.angular.z = 0

			rospy.wait_for_service('/gazebo/set_model_state')
			set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			set_model_state(ModelState(
			    model_name=model_name,
			    pose=model_state.pose,
			    twist=model_state.twist,
			    reference_frame='world'
			))
			rospy.loginfo("Collisions disabled for the drone.")
		except rospy.ServiceException as e:
			rospy.logerr(f"Service call failed: {e}")

	def enable_collisions(self, model_name):
		rospy.wait_for_service('/gazebo/get_model_state')
		try:
			get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
			model_state = get_model_state(model_name, 'world')

			# Reset the twist to re-enable physics
			rospy.wait_for_service('/gazebo/set_model_state')
			set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
			set_model_state(ModelState(
			    model_name=model_name,
			    pose=model_state.pose,
			    twist=Twist(),  # Resetting twist to zero
			    reference_frame='world'
			))
			rospy.loginfo("Collisions enabled for the drone.")
		except rospy.ServiceException as e:
			rospy.logerr(f"Service call failed: {e}")

	def move_drone_to_position(self, target_position, steps=5):
		rospy.wait_for_service('/gazebo/get_model_state')
		
		gt_pose = self.get_gt_pose()
		x = gt_pose.pose.position.x
		y = gt_pose.pose.position.y
		z = gt_pose.pose.position.z

		self.set_state_gazebo_model(
			model=self.drone_name,
			pose=Pose(position=Point(x=target_position.x, y=target_position.y, z=0), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
		)#z=0 because want drone moved along ground

	def reset(self):
		"""
		This is the function called at the start and every time an episode ends to return the drone to its takeoff
		height at (x=0,y=0). The contents of the bellow else statement is more complicated that one might expect
		because the standard gazebo '/gazebo/reset_simulation' and '/gazebo/reset_world' roservice calls caused
		the drone to act weird. You may have to adjust the contents of the else statement to suit your environment.
		"""		
		rospy.loginfo('start reset')

		#First call of reset, arm drone (no need to move to start point as already there):
		if (self.drone.is_armed() == False):
			rospy.loginfo("Arm drone")
			self.drone.arm()
			self.drone.wait_for_status(self.drone.is_armed, True, 2)

			self.go_to_takeoff_height()

			rospy.loginfo("Changing to offboard mode.")
			self.drone.set_mode(PX4_MODE_OFFBOARD)
			rospy.sleep(7)

		else:#all other calls move to start point (no need to arm drone as already armed)
			self.start_point.y = np.random.choice(self.possible_y_start_desired_points)
			self.desired_point.y = self.start_point.y
			self.start_point.x = np.random.choice(self.possible_x_start_points)
			self.desired_point.x = 5 + self.start_point.x

			gt_pose = self.get_gt_pose()        
			x = round(gt_pose.pose.position.x, 2)
			y = round(gt_pose.pose.position.y, 2)
			z = round(gt_pose.pose.position.z, 2)

			rospy.loginfo("Send to x-0.6, y, z")
			cmd = SetPositionWithYawCmdBuilder.build(x=x-0.6, y=y, z=z)
			self.drone.set_pose2d(cmd)
			rospy.sleep(8)

			rospy.loginfo("Get new current x and y position")
			gt_pose = self.get_gt_pose()        
			x = round(gt_pose.pose.position.x, 2)
			y = round(gt_pose.pose.position.y, 2)

			rospy.loginfo("Send to x, y, " + str(self.work_space_z_max + 1.0))
			cmd = SetPositionWithYawCmdBuilder.build(x=x, y=y, z=self.work_space_z_max + 1.0)
			self.drone.set_pose2d(cmd)
			rospy.sleep(4)

			rospy.loginfo("Send to " + str(round(self.start_point.x,2)) + ", " + str(round(self.start_point.y,2)) + ", " + str(self.work_space_z_max + 1.0))
			cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y=self.start_point.y, z=self.work_space_z_max + 1.0)
			self.drone.set_pose2d(cmd)
			rospy.sleep(7)

			self.go_to_takeoff_height()

		self.reset_dist_to_desired_point_and_observations()
		return self._get_obs(), {}

	def go_to_takeoff_height(self):
		rospy.loginfo("Going to takeoff height")
		cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y = self.start_point.y, z = self.start_point.z)
		self.drone.set_pose2d(cmd)
		rospy.sleep(5)

	def reset_yaw_to_zero(self):
		gt_pose = self.get_gt_pose()
		x = gt_pose.pose.position.x
		y = gt_pose.pose.position.y
		z = gt_pose.pose.position.z

		# Build the command with heading (yaw) set to 0
		cmd = SetPositionWithYawCmdBuilder.build(x=x, y=y, z=z, hdg=0)
		self.drone.set_pose2d(cmd)
		rospy.sleep(3)

	def close(self):
		rospy.logdebug("Closing Environment from drone_gazebo.py")
		rospy.signal_shutdown("Closing Environment from drone_gazebo.py")

	def takeoff_drone(self):
		self.drone.takeoff()
		
		rospy.loginfo("Going to takeoff height")
		cmd = SetPositionWithYawCmdBuilder.build(x=self.start_point.x, y=self.start_point.y, z=self.z)
		self.drone.set_pose2d(cmd)
		self.drone.sleep(5)

		rospy.loginfo("Changing to offboard mode.")
		self.drone.set_mode(PX4_MODE_OFFBOARD)
		self.drone.sleep(7.)

	def land_disconnect_drone(self):
		rospy.loginfo("Land drone")
		self.drone.land(block=True)

	def _update_episode(self):
		self.episode_num += 1
		self.cumulated_episode_reward = 0

	def euler_from_quaternion(self, x, y, z, w):
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(t0, t1)

		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.asin(t2)

		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(t3, t4)

		return roll, pitch, yaw #radians
