# Match Additive Manufacturing

This repository contains packages related to additive manufacturing processes. Below is a brief description of each package included in this repository.

## Packages

### 1. ur_trajectory_follower
![ur_trajectory_follower control structure](img/Control_Structure_UR.png)
![ur_trajectory_follower control structure using feedforward instead of control](img/Control_Structure_UR_ff.png)
## Prerequisites

This package requires the following to be installed:

- [ROS](http://wiki.ros.org/ROS/Installation)
- match_mobile_robotics
- [match_lib](https://github.com/pumablattlaus/match_lib_package)

## Installation

To install the packages, clone the repository and build the workspace:

```bash
git clone https://github.com/yourusername/match_additive_manufacturing.git
cd match_additive_manufacturing
catkin_make
```

## Usage

To use the packages, source the workspace and run the desired nodes:

```bash
source devel/setup.bash
```

To launch everything:

```bash
roslaunch parse_mir_path parse_mir_path.launch
roslaunch parse_ur_path parse_ur_path.launch
roslaunch move_mir_to_start_pose move_mir_to_start_pose.launch
# wait for the MIR robot to reach the start pose
roslaunch move_ur_to_start_pose move_ur_to_start_pose.launch

roslaunch ur_trajectory_follower twist_sim.launch
roslaunch ur_trajectory_follower complete_ur_trajectory_follower_ff_only.launch
# or
# roslaunch ur_trajectory_follower complete_ur_trajectory_follower.launch
rosrun topic_tools relay /ur_cmd /mur620/UR10_l/twist_fb_command

roslaunch mir_trajectory_follower mir_trajectory_follower.launch
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

## DOI

https://doi.org/10.5281/zenodo.14507474

## Contact
 
For any questions or inquiries, please contact matchbox@match.uni-hannover.de
