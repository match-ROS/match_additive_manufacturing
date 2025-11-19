# Strand Center App (OAK-D)

This folder contains a minimal DepthAI example `main.py` that detects a filament (strand) by scanning a horizontal image row, samples the aligned depth at the detected pixel, and forwards results as a small JSON message via UDP.

This README explains how to run the app from a host connected to the OAK-D (recommended) and notes about advanced options for device-side examples.

## Files

- `main.py` — DepthAI host script. Captures RGB + depth, finds the filament in a configured image row, and sends JSON via UDP.

## Quick (recommended): run from the host

1. Connect the OAK-D Pro to your host machine via USB.

2. Install Python dependencies on the host (use `--user` or a virtualenv if desired):

```bash
python3 -m pip install --user depthai opencv-python numpy
```

3. Configure UDP destination

- By default `main.py` has configurable constants at the top:

  - `ROW_V` — image row (vertical pixel) scanned for filament (default 360)
  - `UDP_IP` — destination IP (set this to your ROS host IP or receiver)
  - `UDP_PORT` — destination port (default 5005)

Edit these values in `main.py` or modify the script to accept command-line arguments or environment variables if you prefer.

4. Run the script from the repository (or copy it to your working folder):

```bash
# from the workspace root (example path)
cd /home/rosmatch/catkin_ws_recker/src/match_additive_manufacturing/camera_oak/strand-center-app
python3 main.py
```

5. On the receiving end (e.g. your ROS node), listen for UDP JSON messages. Example JSON message structure sent by the script:

```json
{"u": 123.4, "v": 360.0, "depth_mm": 512}
```

The provided ROS node `oak_strand_center_node.py` in `additive_manufacturing_helpers` is designed to receive similar data if configured to forward or to process the camera directly — use the node for direct ROS integration as documented in that package README.

### Launching via the Print GUI

If you're using the Additive Manufacturing GUI (`print_gui/scripts/gui_layout.py`), there is now a **"Launch Strand Center Camera"** button inside the **Setup Functions** panel. Select one or more robots in the left-hand checklist, verify the workspace name, and press the button — the GUI opens a terminal per robot, SSHes into it, sources the workspace, and runs `camera_oak/strand-center-app/main.py`. Use this when you want the robots to run the host script directly without manual SSH steps.

## Running inside a ROS-enabled workspace

If you want ROS topics directly instead of UDP, prefer running the `oak_strand_center_node.py` ROS node (see `additive_manufacturing_helpers/README.md`). Example:

```bash
source /home/rosmatch/catkin_ws_recker/devel/setup.bash
rosrun additive_manufacturing_helpers oak_strand_center_node.py _row_v:=360
```

This node opens the camera and publishes `strand_center/center`, `strand_center/depth_mm`, and `strand_center/status`.

## Advanced: device-side / flashable apps

Most users run a host-side script (above). If you need a permanently resident app running from the device (flashable), look at the Luxonis `depthai-apps` repository which contains examples and instructions for building device-side apps. Flashing or building device-side binaries is an advanced workflow and depends on the example app.

Example starting point:

```bash
git clone https://github.com/luxonis/depthai-apps.git
cd depthai-apps
# follow the repository README for building or flashing specific demos
```

Notes:

- Flashing or running a device-side binary is not required for normal operation. The host script above is simpler and recommended for development, debugging and ROS integration.
- If you need help adapting `main.py` to accept runtime arguments, provide UDP configuration via CLI args, or add a small supervisor/service (systemd) on your host to automatically start it, I can add examples.

## Troubleshooting

- If the script cannot open the device, ensure the OAK is connected and that no other process is holding the camera.
- If depth values are zero or invalid, check that `StereoDepth` is configured and aligned to RGB (the provided pipeline aligns depth to RGB by default).
- If you do not see UDP packets at the receiver, verify firewall rules and that `UDP_IP`/`UDP_PORT` are reachable (try `nc -u -l 5005` on the receiving host to listen).

## License

This file follows the repository license. Use and adapt per project licensing.

---

If you want, I can:

- Add CLI argument support to `main.py` (UDP IP/port, row_v) and update this README with usage examples.
- Add a small systemd unit file to auto-start the script on a dedicated host machine.

Tell me which of those you'd like and I'll implement it.
