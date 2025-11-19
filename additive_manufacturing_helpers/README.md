# Additive Manufacturing Helpers

## OAK-D Strand Center Node

`oak_strand_center_node.py` streams detections from an OAK-D Pro camera that runs the strand-center DepthAI app. It analyses the RGB preview on the host, samples the depth image at the detected pixel, and publishes the result as ROS topics.

### Published topics

- `strand_center/center` (`geometry_msgs/PointStamped`): Pixel column (`x`), row (`y`), and aligned depth in millimetres (`z`). The header stamp corresponds to the processing time.
- `strand_center/depth_mm` (`std_msgs/Float32`): Depth at the detected pixel in millimetres.
- `strand_center/status` (`std_msgs/String`): Simple heartbeat (`detected` / `no_detection`).

### Parameters

| Name | Default | Description |
|------|---------|-------------|
| `~row_v` | `360` | Image row (pixel) used for filament search. |
| `~publish_json` | `false` | If `true`, embeds the latest detection as JSON inside the status topic for debugging. |
| `~forward_udp_ip` | unset | When set, forwards detections as JSON via UDP to the given host (port `~forward_udp_port`, default `5005`). |
| `~debug_log` | `true` | Enables verbose debug logging.

### Requirements

Install the DepthAI Python runtime on the host that is connected to the OAK device:

```bash
python3 -m pip install depthai opencv-python numpy
```

### Usage

```bash
source /home/rosmatch/catkin_ws_recker/devel/setup.bash
rosrun additive_manufacturing_helpers oak_strand_center_node.py _row_v:=360
```

Adjust `_row_v` and UDP parameters to match your camera app.
