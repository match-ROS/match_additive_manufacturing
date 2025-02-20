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
- `get_total_Pose.py`: Calculates the total pose of the UR robot ee in world coordinates from the base_link pose and the pose of the mobile platform.
- `ur_path_direction_controller.py`: Calculates the direction to the next waypoint and multiplies it with the feed rate to get the velocity command for the UR robot. The z-component of the velocity is controlled by PID.

## Future Work
- publish ur_pose in world
- calculate error in world
- control velocity from error and target velocity in world
- transform control velocity to base_link
- Add launch files for easy startup
- Implement Python scripts for trajectory generation and control
