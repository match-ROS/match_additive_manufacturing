# Parse UR Path

This ROS package reads the UR's target path from a given python file.
The path should be in a form:

```
def toolX():
    return [51.615323,51.595286,51.575249, ...]

def toolY():
    return [39.752500,39.752500,39.752500, ...]
``` 

## Features
- Reads the path from a python file
- Transforms the path given a transformation (r_x,r_y, ...)


## Usage
 ```roslaunch parse_ur_path parse_ur_path.launch```

## Topics

<!-- ### Input Topics
- `/ur_trajectory_follower/goal_path` (type: `nav_msgs/Path`): The path that the UR robot should follow.
- `/ur_trajectory_follower/feed_rate` (type: `std_msgs/Float64`): The current feed rate setting of the UR robot in world coordinates in m/s.
- `/ur_trajectory_follower/lateral_nozzle_pose_override` (type: `double`): Setting to control the nozzle height while printing. -->

### Output Topics
- `/ur_path_original` (type: `nav_msgs/Path`): Original path as read from the input file.
- `/ur_path_transformed` (type: `nav_msgs/Path`): Path from input file with transformation applied.


## Parameters
- `~tx`,`~ty`,`~tz` (type: `double`, default: `0.0`): Translation in x,y,z
- `~rx`,`~ry`,`~rz` (type: `double`, default: `0.0`): Rotation in x,y,z




