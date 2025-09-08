# laser_scanner_tools

Utilities for working with the Keyence laser scanner.

## Node: pointcloud_assembler

Aggregates scans into a single PointCloud2 in a target frame with start/stop services.

- Subscribes:
  - `~input_topic` (PointCloud2 or LaserScan) — default `/scan` for LaserScan or `/profiles` for PointCloud2
- Publishes:
  - `~output_topic` (sensor_msgs/PointCloud2), latched — default `/assembled_pointcloud`
- Services:
  - `~start_acquisition` (std_srvs/Trigger)
  - `~stop_acquisition` (std_srvs/Trigger)
- Parameters:
  - `~input_topic` (string, default `/scan`)
  - `~input_type` (string: `pointcloud2`|`laserscan`, default `pointcloud2`)
  - `~target_frame` (string, default `mur620c/base_footprint`)
  - `~source_frame` (string, default empty; override source frame if message header is incorrect)
  - `~output_topic` (string, default `assembled_pointcloud`)
  - `~max_points` (int, default 5,000,000)

### Launch

```
roslaunch laser_scanner_tools pointcloud_assembler.launch \
  input_topic:=/profiles \
  input_type:=pointcloud2 \
  target_frame:=mur620c/base_footprint \
  source_frame:="" \
  output_topic:=/assembled_pointcloud
```

Start and stop acquisition:

```
rosservice call /pointcloud_assembler/start_acquisition
rosservice call /pointcloud_assembler/stop_acquisition
```

### Notes
- Requires TF between the scanner frame (e.g. `sensor_optical_frame`) and the target frame (e.g. `mur620c/base_footprint`).
- If using `input_type:=laserscan`, ensure the `laser_geometry` package is installed and the input topic carries `sensor_msgs/LaserScan` messages.
