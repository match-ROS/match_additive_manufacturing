#!/bin/bash

sudo apt install python3-requests -y

# NOT available for Ros-One:
# sudo apt install ros-$ROS_DISTRO-dynamixel-sdk
# sudo apt install ros-$ROS_DISTRO-dynamixel-workbench

cd ..
git clone https://github.com/pumablattlaus/match_lib_package.git
cd match_lib_package
git submodule update --init --recursive

cd ..
git clone -b noetic-devel https://github.com/match-ROS/match_mobile_robotics.git
cd match_mobile_robotics
./setup_full.sh