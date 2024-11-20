# UR Trajectory Follower

This ROS package allows the UR robot to follow a given path to print while driving. For this, the movement of the mobile platform has to be taken into account.


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
- `/ur_trajectory_follower/feed_rate` (type: `std_msgs/Float64`): The current feed rate setting of the UR robot in world coordinates in m/s. (Or as factor of the path velocity)
- `/ur_trajectory_follower/lateral_nozzle_pose_override` (type: `double`): Setting to control the nozzle height while printing.

### Output Topics
- `/ur_trajectory_follower/k_idx` (type: `std_msgs/Int32`): The index of the current waypoint in the path.

## Services
- `/ur_trajectory_follower/start` (type: `std_srvs/Trigger`): Service to start following the path.
- `/ur_trajectory_follower/stop` (type: `std_srvs/Trigger`): Service to stop following the path.

## Parameters
- `~feed_rate` (type: `double`, default: `0.1`): Target velocity of the UR robot in world coordinates in m/s. (Or dt)
- `~lateral_nozzle_pose` (type: `double`, default: `0.1`): Pose of nozzle to ur path.


## Scripts
- `ur_vel_induced_by_mir.py`: Calculates the velocity of the UR robot induced by the movement of the mobile platform.
- `combine_twists.py`: Combines a list of twists into a single twist, to be used as a velocity command for the UR robot.
- `orthogonal_error_correction.py`: Takes a normalized vector orthogonal to the path (normal) as well as goal pose and current pose of the ur nozzle and calculates a twist for a control to follow the path more accurately.

## Future Work
- Add launch files for easy startup
- Implement Python scripts for trajectory generation and control
