# foam_volume_flow_sensor

Bridge the dual 4–20 mA volume-flow sensors that are sampled by the Arduino Mega sketch in `arduino/volume_fow.ino` into ROS topics.

## Arduino firmware

1. Flash `arduino/volume_fow.ino` to the Mega 2560.
2. Keep the CSV header line (`time_ms,chan,...`) – the ROS node simply skips it.
3. (Optional) Add a prefix like `FLOW,` before each data row. Then set the node parameter `line_prefix:=FLOW,` to ignore any boot messages on the serial port.

## ROS dependencies

- `rospy`, `std_msgs` (already in the package manifest).
- `pyserial` for accessing `/dev/ttyACM*` ports: `pip install pyserial` (in the same environment you run ROS).

After adding the package, rebuild your workspace:

```bash
cd ~/catkin_ws_recker
catkin_make
source devel/setup.bash
```

## Custom message

`msg/FlowSample.msg` exposes:

| Field | Description |
| --- | --- |
| `header.stamp` | Timestamp from the Arduino `millis()` counter |
| `header.frame_id` | Set via `frame_id` param (default `flow_sensor`) |
| `channel` | Sensor channel (1 or 2) |
| `raw_adc` | Averaged ADC reading |
| `voltage_v` | Converted shunt voltage |
| `current_ma` | Filtered loop current in milliamps |
| `percent` | Normalised 4–20 mA percentage |
| `engineering_value` | Optional engineering unit (NaN if disabled on Arduino) |

## Launching

Use the provided launch file to bring up the serial bridge:

```bash
roslaunch foam_volume_flow_sensor flow_serial_bridge.launch port:=/dev/ttyACM0
```

Launch arguments / node parameters:

- `port` – Serial device path (default `/dev/ttyACM0`).
- `baud` – Baud rate, must match the Arduino sketch (default `115200`).
- `topic` – Topic for `FlowSample` messages (default `foam_volume_flow_sensor/samples`).
- `frame_id` – Frame ID stored in `header.frame_id`.
- `line_prefix` – Optional string the line must start with before parsing.
- `timeout` – Serial timeout (seconds) to detect cable disconnects.

The node will automatically reconnect if the USB cable is unplugged/replugged.

## Consuming the data

Subscribe to the topic specified by `topic`, e.g.:

```bash
rostopic echo /foam_volume_flow_sensor/samples
```

Each message reports the smoothed current for both channels in round-robin order (every ~100 ms per channel with the default Arduino firmware). Use standard ROS tools or your own nodes to log, visualise, or react to these flow readings.
