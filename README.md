# Efficient-UAV-DRL-OA-Gym-Gazebo
## Overview 
#### The code in this repository is connected to the yet to be published paper 'FERO: Efficient Deep Reinforcement Learning based UAV Obstacle Avoidance at the Edge'.
#### The bellow steps provide instructions to 1) setup ROS/Gym/Gazebo environment on a Ubuntu 20.04 machine (e.g., desktop), 2) run python/pytorch files to train D3QN/SAC baseline/TL models and python files to convert pytorch to half precision onnx, 3) setup the nvidia jetson orin nano or nvidia jetson nano to run the models (as TensorRT engines)
## 1: Setup ROS/Gym/Gazebo environment on a Ubuntu 20.04 machine (e.g., desktop)
### Step 1.1 - Ensure you have a Ubuntu 20.04 Machine and install ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)
### Step 1.2 - Run various updates/installs and create a catkin workspace:
```
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
sudo snap install sublime-text --classic
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```
### Step 1.3 - Get the Gazebo Model for the Uvify IFO-S (https://github.com/decargroup/ifo_gazebo):
#### Step 1.3a - Execute the following commands:
```
cd ~/catkin_ws/src
git clone https://github.com/decarsg/ifo_gazebo.git --recursive
cd ..
catkin config --blacklist px4
catkin build
catkin build
cd ..
bash ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot/Tools/setup/ubuntu.sh
```
#### Step 1.3b - Relogin or reboot and execute the following commands:
```
sudo apt install python3-pip
```
```
pip3 install pyulog
pip3 install future
sudo apt upgrade -y
```
```
cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make distclean

cd ~
pip3 install --user empy
pip3 install --user packaging
pip3 install --user toml
pip3 install --user numpy
pip3 install --user jinja2

cd ~/catkin_ws/src/ifo_gazebo/PX4-Autopilot
make px4_sitl gazebo
#if gazebo black screen then ctrl+c and run make command again
```
```
#ctrl+c
cd ~/catkin_ws/src/ifo_gazebo
rm -r real*
git clone https://github.com/pal-robotics/realsense_gazebo_plugin.git
cd ~/catkin_ws
catkin build
#run catkin build again if previous catkin build returns with a warning
```
#### Step 1.3c - execute more commands:
```
cd ~
nano ubuntu_sim_ros_noetic.sh
#fill ubuntu_sim_ros_noetic.sh with the contents of https://gist.githubusercontent.com/ekaktusz/a1065a2a452567cb04b919b20fdb57c4/raw/8be54ed561db7e3a2ce61c9c7b1fb9fec72501f4/ubuntu_sim_ros_noetic.sh
#exit and save ubuntu_sim_ros_noetic.sh
bash ubuntu_sim_ros_noetic.sh
#answer 'y' for any prompts
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/src/ifo_gazebo/setup_ifo_gazebo.bash suppress" >> ~/.bashrc
cd ~/catkin_ws
source ~/.bashrc
roslaunch ifo_gazebo ifo_empty_world_1.launch
```
### Step 1.4 - Get the ROS package that allows a user to communicate with PX4 autopilot using MAVROS by executing the following commands (based off https://github.com/troiwill/mavros-px4-vehicle):
```
#cnrl+c
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash
```
### Step 1.5 - Install scipy, gym and torch:
```
pip3 install scipy
pip3 install gym==0.21
pip3 install torch
```
### Step 1.6 - Move the files from this github repository into the appropriate places as outlined bellow
#### Step 1.6a - Add python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 1.6ai - Download python scripts in 'desktop_files' folder and drag and drop files into '~/catkin_ws/src/mavros-px4-vehicle/scripts' folder
##### Step 1.6aii - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```
#### Step 1.6b - Replace 3 of your launch files
##### Step 1.6bi - Download the '.launch' files in 'desktop_files' folder
##### Step 1.6bii - Replace the files with the same names in '~/catkin_ws/src/ifo_gazebo/ifo_gazebo/launch/' with these new files
#### Step 1.6c - Create a few folders and add world files to one of them
##### Step 1.ci -  Create folders at '~/catkin_ws/src/mavros-px4-vehicle'
```
cd ~/catkin_ws/src/mavros-px4-vehicle
mkdir {models,plots,worlds}
```
##### Step 1.cii - Download the '.world' files in the 'desktop_files' folder and move them to newly created 'worlds' folder

## 2: Run python/pytorch files to train models or convert pytorch to half precision onnx
### Step 2.1 - Launch world (you may have to restart machine before this step)
```
cd ~
source ~/.bashrc
roslaunch ifo_gazebo drone_and_world.launch
```
### Step 2.2 - Run relevant python files
#### Step 2.2a - Move to catkin workspace and source bashrc file
```
cd ~/catkin_ws
source ~/.bashrc
```
#### Step 2.2b - if training D3QN/SAC model with 32by6 state space
```
rosrun mavros_px4_vehicle train_main_d3qn.py #sac_main.py if SAC
```
#### Step 2.2c - if training D3QN/SAC model with 106by80 state space, remove _106by80 in all files that end in this and run as in 2.2b
#### Step 2.2d - if training D3QN/SAC model with 66by1 state space, remove _66by1 in all files that end in this and run as in 2.2b
#### Step 2.2e - if training transfer learning model remove _TL... in all files that end in this and run as in 2.2b
#### Step 2.2f - if converting to half precision onnx, run the bellow commands
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
python3 pytorch_to_onnx.py #if 106by80 or 66by1 remove this extension then run
python3 onnx_to_onnxHalf.py
```

## 3: Setup the nvidia jetson orin nano or nvidia jetson nano to run the models (as TensorRT engines)
### Step 3.1 - Install Ubuntu 20.04 and ROS Noetic on jetson orin nano/jetson nano (for jetson nano install ubuntu 20.04 by following https://qengineering.eu/install-ubuntu-20.04-on-jetson-nano.html)
### Step 3.2 - Execute the following commands:
```
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
roscore
#cntrl+c
sudo apt update
sudo apt-get install python3-catkin-tools -y
sudo apt install git
```
### Step 3.3 - Create catkin workspace:
```
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin build
source devel/setup.bash
echo $ROS_PACKAGE_PATH
```
### Step 3.4 - More commands:
```
cd ~
nano ubuntu_sim_ros_noetic.sh
#fill ubuntu_sim_ros_noetic.sh with the contents of https://gist.githubusercontent.com/ekaktusz/a1065a2a452567cb04b919b20fdb57c4/raw/8be54ed561db7e3a2ce61c9c7b1fb9fec72501f4/ubuntu_sim_ros_noetic.sh
#exit and save ubuntu_sim_ros_noetic.sh
bash ubuntu_sim_ros_noetic.sh
#answer 'y' for any prompts
```
### Step 3.5 - Get the ROS package that allows a user to communicate with PX4 autopilot using MAVROS by executing the following commands (based off https://github.com/troiwill/mavros-px4-vehicle):
```
#cnrl+c
cd ~/catkin_ws/src
git clone https://github.com/troiwill/mavros-px4-vehicle.git
chmod +x mavros-px4-vehicle/scripts/*.py
chmod +x mavros-px4-vehicle/test/*.py
ln -s mavros-px4-vehicle mavros-px4-vehicle
cd ~/catkin_ws
catkin build
source devel/setup.bash
```
### Step 3.6 - Install scipy, gym, torch and re-install numpy so it v1.21:
```
pip3 install scipy
pip3 install gym
pip3 install torch
pip3 uninstall numpy
pip3 install numpy=1.21
```
### Step 3.7 - Move the files from this github repository into the appropriate places as outlined bellow
#### Step 3.7a - Add python scripts to '~/catkin_ws/src/mavros-px4-vehicle/scripts'
##### Step 3.7ai - Download python scripts in 'jetson_files' folder and drag and drop files into '~/catkin_ws/src/mavros-px4-vehicle/scripts' folder
##### Step 3.7aii - Ensure the python scripts are executable
```
cd ~/catkin_ws/src/mavros-px4-vehicle/scripts
chmod +x *.py
```

### Step 3.8 - onnx file to TensorRT engine:
#### Step 3.8a - change to the scripts folder directory and run the bellow command
```
python3 onnxHalf_to_trtHalf_orin_nano.py #onnxHalf_to_trtHalf_nano.py if jetson nano
```

### Step 3.9 - Run TRT engines on jetson orin nano/jetson nano:
#### Step 3.9a - Setup LAN setup as shown in paper
#### Step 3.9b - On Desktop where 192.168.8.107:11311 will be different for your machine (run ifconfig):
```
edit .bashrc file at ~ 
export ROS_MASTER_URI=http://192.168.8.107:11311
export ROS_IP=192.168.8.107
sudo reboot
source ~/.bashrc
roslaunch ifo_gazebo drone_and_world.launch
```
#### Step 3.9c - On jetson orin nano/jetson nano where 192.168.8.134 will be different for you jetson device (run ifconfig):
```
edit .bashrc file at ~ 
export ROS_MASTER_URI=http://192.168.8.107:11311
export ROS_IP=192.168.8.134
sudo reboot
source ~/.bashrc
rosrun mavros_px4_vehicle eval_main_d3qn_trt_orin_nano.py #or eval_main_d3qn_trt_nano.py (remove '_106by80' in file 'drone_gym_gazebo_env_discrete_106by80.py' if have this state space)
```
