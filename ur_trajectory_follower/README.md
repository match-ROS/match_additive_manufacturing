# UR Trajectory Follower

This ROS package allows a UR robot to follow a given path. The package is designed to be flexible and easy to use, with launch files and Python scripts to be generated in the future.

## Features
- Follow a predefined path with a UR robot
- Easy integration with other ROS packages

## Installation
Clone this repository into your catkin workspace and build the package:
```sh
cd ~/catkin_ws/src
git clone <repository_url>
cd ~/catkin_ws
catkin_make
```

## Usage
To be updated with launch file and Python script usage.

## Topics

### Input Topics
- `/ur_trajectory_follower/goal_path` (type: `nav_msgs/Path`): The path that the UR robot should follow.
- `/ur_trajectory_follower/feed_rate` (type: `std_msgs/Float64`): The current feed rate setting of the UR robot in world coordinates in m/s.
- `/ur_trajectory_follower/lateral_nozzle_pose_override` (type: `double`): Setting to control the nozzle height while printing.

### Output Topics
- `/ur_trajectory_follower/k_idx` (type: `std_msgs/Int32`): The index of the current waypoint in the path.

## Services
- `/ur_trajectory_follower/start` (type: `std_srvs/Trigger`): Service to start following the path.
- `/ur_trajectory_follower/stop` (type: `std_srvs/Trigger`): Service to stop following the path.

## Parameters
- `~feed_rate` (type: `double`, default: `0.1`): Target velocity of the UR robot in world coordinates in m/s.
- `~lateral_nozzle_pose` (type: `double`, default: `0.1`): Pose of nozzle to ur path.

## Future Work
- Add launch files for easy startup
- Implement Python scripts for trajectory generation and control
