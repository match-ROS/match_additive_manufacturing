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

### Basic Parameters
- `~lookahead_distance` (default: 0.1): Base lookahead distance for pure pursuit
- `~distance_threshold` (default: 0.02): Tangential distance threshold for waypoint completion  
- `~search_range` (default: 5): Number of points to search for lookahead point
- `~Kv` (default: 1.0): Linear velocity multiplier
- `~control_rate` (default: 100): Control loop frequency in Hz
- `~dT` (default: 0.5): Time step for velocity calculations

### Advanced High-Speed Tracking Parameters (RL Version)
- `~max_lookahead_distance` (default: 0.5): Maximum adaptive lookahead distance
- `~lookahead_velocity_gain` (default: 0.1): Scaling factor for velocity-based lookahead adaptation
- `~K_idx` (default: 0.05): Index error compensation gain (increased from 0.01 for better high-speed tracking)  
- `~K_feedforward` (default: 0.2): Feedforward compensation gain for predictive control
- `~K_distance` (default: 0.0): Distance error multiplier
- `~K_orientation` (default: 0.5): Orientation error multiplier

## Nodes
### mir_trajectory_follower_pure_pursuit_RL (Reinforcement Learning Enhanced)
- **File:** [mir_trajectory_follower_pure_pursuit_RL.py](scripts/mir_trajectory_follower_pure_pursuit_RL.py)
- **Description:** Advanced trajectory follower with reinforcement learning enhancements and high-speed tracking improvements. Features adaptive lookahead distance, enhanced lag compensation, and feedforward control for better performance at high speeds.
- **Key Improvements:**
  - **Adaptive Lookahead:** Dynamically adjusts lookahead distance based on current velocity (0.1m to 0.5m)
  - **Enhanced Lag Compensation:** Non-linear index error compensation for significant tracking lags
  - **Feedforward Control:** Predictive velocity compensation to anticipate trajectory requirements  
  - **High-Speed Optimization:** Reduces lag between mir_target_pose and actual robot pose at high speeds

### mir_trajectory_follower_pure_pursuit
- **File:** [mir_trajectory_follower_pure_pursuit.py](scripts/mir_trajectory_follower_pure_pursuit.py)
- **Description:** Standard pure pursuit implementation. Follows a trajectory using the Pure Pursuit algorithm. It processes incoming path data, identifies a lookahead point, computes curvature, and publishes velocity commands.
- **TF Broadcasting:** Publishes transforms for the current pose, target (lookahead) point, and current position for debugging.

### mir_trajectory_follower (Custom Path Follower)
- ***File:*** [mir_trajectory_follower.py](scripts/mir_trajectory_follower.py)
- ***Description:*** Implements a custom control loop to follow a given trajectory based on distance and angular alignment control.
- ***Additional Topics:*** Subscribes to a start command (`/start_follow_path`) and publishes path completion status to `/path_following_complete`.

## Launch Files
- [mir_trajectory_follower.launch](launch/mir_trajectory_follower.launch): Launches the pure pursuit node with default parameter settings.
- Custom remapping or parameter changes can be done by editing the launch file or using command-line arguments with `roslaunch`.