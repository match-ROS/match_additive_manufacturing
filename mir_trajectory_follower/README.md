# Parse MiR Path

This ROS package reads the MiR's target path from a given python file.
The path should be in a form:

```
def mirX():
    return [51.615323,51.595286,51.575249, ...]

def mirX():
    return [39.752500,39.752500,39.752500, ...]
``` 

## Features
- Reads the path from a python file
- Transforms the path given a transformation (r_x,r_y, ...)


## Usage
 ```roslaunch parse_mir_path parse_mir_path.launch```

## Topics

<!-- ### Input Topics
- `/ur_trajectory_follower/goal_path` (type: `nav_msgs/Path`): The path that the UR robot should follow.
- `/ur_trajectory_follower/feed_rate` (type: `std_msgs/Float64`): The current feed rate setting of the UR robot in world coordinates in m/s.
- `/ur_trajectory_follower/lateral_nozzle_pose_override` (type: `double`): Setting to control the nozzle height while printing. -->

### Output Topics
- `/mir_path_original` (type: `nav_msgs/Path`): Original path as read from the input file.
- `/mir_path_transformed` (type: `nav_msgs/Path`): Path from input file with transformation applied.


## Parameters
- `~tx`,`~ty`,`~tz` (type: `double`, default: `0.0`): Translation in x,y,z
- `~rx`,`~ry`,`~rz` (type: `double`, default: `0.0`): Rotation in x,y,z




