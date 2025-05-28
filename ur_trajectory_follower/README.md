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
To start the UR trajectory follower with feed_rate only in print direction and orthogonal error correction, run the following command:
```sh
roslaunch ur_trajectory_follower complete_ur_trajectory_follower_ff_only.launch
```

To start the UR trajectory follower with feed_rate in all directions and _no_ orthogonal error correction, run the following command:
```sh
roslaunch ur_trajectory_follower complete_ur_trajectory_follower.launch
```
### HW
Topics etc. have to be adjusted. For Exp. for the compensation:
```sh
roslaunch ur_trajectory_follower compensate_mir.launch ur_cmd_vel_local_topic:="/mur620d/UR10_r/twist_controller/command_collision_free" robot_name:=mur620d mir_odom_topic:="mur620d/odom"
```

## Topics

### Input Topics
- `/ur_trajectory_follower/ur_path_transformed` (type: `nav_msgs/Path`): The path that the UR robot should follow.
- `/ur_trajectory_follower/velocity_override` (type: `std_msgs/Float64`): The current feed rate setting of the UR robot as factor of the path velocity [%]
- `/ur_trajectory_follower/nozzle_height_override` (type: `double`): Setting to control the nozzle height while printing [m]
- `/current_nozzle_pose` (type: `geometry_msgs/PoseStamped`): The current pose of the UR robot.

### Output Topics
- `/ur_trajectory_follower/path_index` (type: `std_msgs/Int32`): The index of the current waypoint in the path.

## Services
- `/ur_trajectory_follower/start` (type: `std_srvs/Trigger`): Service to start following the path.
- `/ur_trajectory_follower/stop` (type: `std_srvs/Trigger`): Service to stop following the path.

## Parameters
- `~nozzle_height_default` (type: `double`, default: `0.1`): Default height of nozzle to ur path.
- `~kp_z` (type: `double`, default: `1.0`): Proportional gain for PID controller for z-component of velocity.
- `~ki_z` (type: `double`, default: `0.0`): Integral gain for PID controller for z-component of velocity.
- `~kd_z` (type: `double`, default: `0.0`): Derivative gain for PID controller for z-component of velocity.


## Scripts
- `ur_vel_induced_by_mir.py`: Calculates the velocity of the UR robot induced by the movement of the mobile platform.
- `combine_twists.py`: Combines a list of twists into a single twist, to be used as a velocity command for the UR robot.
- `orthogonal_error_correction.py`: Takes a normalized vector orthogonal to the path (normal) as well as goal pose and current pose of the ur nozzle and calculates a twist for a control to follow the path more accurately.
    - this script could be used for any direction correction, not only orthogonal to the path by setting the normal vector accordingly.
- `path_idx_advancer.py`: Determines the index of the next waypoint in the path from the current pose of the UR robot. There are 3 different methods implemented:
    - `radius`: Determines the index of the next waypoint based on the distance to the goal pose.
    - `collinear`: Determines the index of the next waypoint based on the distance to the goal pose collinear to the path between the last and the next waypoint.
    - `virtual line`: Determines the index of the next waypoint based on the distance to a virtual line between the last and the next waypoint. The orientation of the line is determined by the angle bisector of the inccoming and outgoing segment of the next waypoint.
- `ur_path_direction_controller.py`: Calculates the direction to the next waypoint and multiplies it with the feed rate to get the velocity command for the UR robot. If _ff\_only_ is set, the direction is only calculated based on the path. Otherwise, the direction is from the current pose to the next waypoint. If _ff\_only_ is set, consider using orthogonal_error_correction.py to correct the direction. The z-component of the velocity is controlled by PID.
- `rotate_twist_by_pose.py`: Rotates a twist by a given pose. This is used to transform the velocity command from world to base_link coordinates.

## Future Work
- orthogonal_error_correction.py: Implement a PID controller for the orthogonal error correction.
    - Or maybe new node for PID from twist error to twist command

# Test
```sh
roslaunch match_gazebo scale.launch
roslaunch mur_launch_sim mur_620.launch enable_dual_collison_avoidance:=false robot_x:=51.61532 robot_y:=39.752500
roslaunch ur_trajectory_follower twist_sim.launch

roslaunch ur_trajectory_follower complete_ur_trajectory_follower.launch

rosrun topic_tools relay /ur_cmd /mur620/UR10_l/twist_fb_command 
```


## Before
```sh
#roslaunch parse_mir_path parse_mir_path.launch
roslaunch parse_ur_path parse_ur_path.launch
#roslaunch move_mir_to_start_pose move_mir_to_start_pose.launch
roslaunch move_ur_to_start_pose move_ur_to_start_pose.launch
```