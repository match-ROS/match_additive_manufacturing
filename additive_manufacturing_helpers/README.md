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

## Running the app on the camera (OAK-D)

you run a DepthAI pipeline from the host which loads the pipeline and network to the camera. Below are two practical ways to run the strand-center app.

1) Quick (recommended) — run from the host

- Connect the OAK-D Pro to the host via USB.
- Install DepthAI and dependencies on the host (use --user if you prefer a local install):

```bash
python3 -m pip install --user depthai opencv-python numpy
```

- Source your catkin workspace and launch the ROS node included in this package; this node will open the camera, load the strand-center pipeline onto the device, read detections and publish the ROS topics listed above:

```bash
source /home/rosmatch/catkin_ws_recker/devel/setup.bash
rosrun additive_manufacturing_helpers oak_strand_center_node.py _row_v:=360
```

Adjust `_row_v` and the UDP forwarding parameters (if used) to match your camera app.

2) Advanced — use Luxonis DepthAI example apps / flashable device apps

- If you prefer to run one of the example apps or want to build/flash a device-side binary, see the DepthAI apps repository and its README for build and flash instructions:

```bash
git clone https://github.com/luxonis/depthai-apps.git
cd depthai-apps
# follow the repo README for building or running specific demos
```

- Note: most workflows do not require flashing; running the host-side script is simpler and recommended for development and ROS integration.

If you tell me which strand-center script or repo you're using, I can give exact commands for that example or add a link to the specific repo in this README.
