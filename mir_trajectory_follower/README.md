# mir_trajectory_follower

The `mir_trajectory_follower` package provides nodes to follow a trajectory for a MIR mobile robot. It offers two trajectory-following implementations:
  
- **Custom Path Follower:** Uses a control loop based on distance and angular error.
- **Pure Pursuit:** Implements a pure pursuit algorithm to compute lookahead points and steering commands.

Both nodes subscribe to a desired path (as a `nav_msgs/Path`) and the robot's current pose while publishing velocity commands (`geometry_msgs/Twist`) for path following.

## Features

- **Path Following:** Continuously monitor the current robot pose and adjust velocity commands to follow the given path.
- **Pure Pursuit:** Compute a lookahead point based on a configurable lookahead distance and search range.

## Usage
To launch the pure pursuit node, use the provided launch file:
```bash
roslaunch mir_trajectory_follower mir_trajectory_follower.launch
```

## Parameters
You can override the default parameters using the launch file or command line switches. Here are some of the key parameters:

- **mir_path_topic**: Topic for the input path (default: `/mir_path_original`).
- **mir_pose_topic**: Topic for the robot pose (default: `/mur620/mir_pose_simple`).
- **cmd_vel_topic**: Topic to publish velocity commands (default: `/mur620/mobile_base_controller/cmd_vel`).
- **trajectory_index_topic**: Topic for trajectory indexing (default: `/trajectory_index`).

Other tuning parameters include lookahead distance, distance threshold, and control gains:

- `~lookahead_distance` (default: 0.25)
- `~distance_threshold` (default: 0.25)
- `~search_range` (default: 20)
- ```~Kv``` (default: 1.0)
- `~control_rate` (default: 100)
- `~dT` (default: 0.2)

## Nodes
### mir_trajectory_follower_pure_pursuit
- **File:** [mir_trajectory_follower_pure_pursuit.py](scripts/mir_trajectory_follower_pure_pursuit.py)
- **Description:** Follows a trajectory using the Pure Pursuit algorithm. It processes incoming path data, identifies a lookahead point, computes curvature, and publishes velocity commands.
- **TF Broadcasting:** Publishes transforms for the current pose, target (lookahead) point, and current position for debugging.

### mir_trajectory_follower (Custom Path Follower)
- ***File:*** [mir_trajectory_follower.py](scripts/mir_trajectory_follower.py)
- ***Description:*** Implements a custom control loop to follow a given trajectory based on distance and angular alignment control.
- ***Additional Topics:*** Subscribes to a start command (`/start_follow_path`) and publishes path completion status to `/path_following_complete`.

## Launch Files
- [mir_trajectory_follower.launch](launch/mir_trajectory_follower.launch): Launches the pure pursuit node with default parameter settings.
- Custom remapping or parameter changes can be done by editing the launch file or using command-line arguments with `roslaunch`.