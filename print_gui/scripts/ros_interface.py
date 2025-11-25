import os
import subprocess
import yaml
import threading
from typing import Optional
import rospy
from geometry_msgs.msg import PoseStamped
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import QTimer
import rospy
import tf.transformations as tf_trans
from rosgraph_msgs.msg import Log
from std_msgs.msg import Float32, Int32, Int16
from sensor_msgs.msg import BatteryState
from rosgraph_msgs.msg import Log
import rosnode

import rospy
from geometry_msgs.msg import PoseStamped


DEBUG_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "rosconsole_debug.config")
)


def _is_debug_enabled(gui) -> bool:
    return bool(getattr(gui, "is_debug_enabled", lambda: False)())


def _build_debug_env(gui, base_env=None):
    env = os.environ.copy() if base_env is None else base_env.copy()
    if _is_debug_enabled(gui):
        env["ROSCONSOLE_CONFIG_FILE"] = DEBUG_CONFIG_PATH
    else:
        env.pop("ROSCONSOLE_CONFIG_FILE", None)
    return env


def _remote_debug_prefix(gui, workspace: Optional[str] = None) -> str:
    workspace = (workspace or "").strip()
    if not (_is_debug_enabled(gui) and workspace):
        return ""
    remote_config = f"~/{workspace}/src/match_additive_manufacturing/print_gui/config/rosconsole_debug.config"
    return f"ROSCONSOLE_CONFIG_FILE={remote_config}; "


def _popen_with_debug(command, gui, shell=False, **kwargs):
    env = _build_debug_env(gui, kwargs.pop("env", None))
    return subprocess.Popen(command, shell=shell, env=env, **kwargs)


def _run_remote_commands(
    gui,
    description: str,
    commands,
    *,
    use_workspace_debug: bool = False,
    target_robots=None,
):
    """Run commands over SSH for each target robot."""

    selected_robots = target_robots if target_robots is not None else gui.get_selected_robots()
    if isinstance(selected_robots, str):
        selected_robots = [selected_robots]
    if not selected_robots:
        print(f"No robots selected. Skipping {description}.")
        return

    cmd_list = list(commands)
    if not cmd_list:
        print(f"No commands provided for {description}.")
        return

    workspace = (gui.get_workspace_name() or "").strip()
    debug_prefix = _remote_debug_prefix(gui, workspace) if use_workspace_debug else ""
    env_prefix = [
        "source ~/.bashrc",
        "export ROS_MASTER_URI=http://roscore:11311/",
        "source /opt/ros/noetic/setup.bash",
    ]
    if workspace:
        env_prefix.append(f"source ~/{workspace}/devel/setup.bash")

    for robot in selected_robots:
        per_robot_cmds = []
        for cmd in cmd_list:
            resolved_cmd = cmd(robot) if callable(cmd) else cmd
            if not isinstance(resolved_cmd, str):
                raise ValueError("Commands must resolve to strings")
            resolved_cmd = resolved_cmd.strip()
            if debug_prefix:
                resolved_cmd = f"{debug_prefix}{resolved_cmd}"
            per_robot_cmds.append(resolved_cmd)

        remote_cmd = "; ".join(env_prefix + per_robot_cmds)
        ssh_cmd = ["ssh", "-t", "-t", robot, remote_cmd]
        print(f"{description} on {robot} with command: {remote_cmd}")
        _popen_with_debug(ssh_cmd, gui)


def _format_ros_list_arg(values):
    """Format Python list as double-quoted string literal for roslaunch args."""
    return '"' + str(values).replace("'", '"') + '"'

class ROSInterface:
    def __init__(self, gui):
        self.gui = gui
        self.updated_poses = {}
        self.virtual_object_pose = None
        self.battery_states = {}
        self.active_battery_subs = set()
        self.current_index = 0

        # Rosbag config
        self.rosbag_topics = [
            "/tf",
            "/ur_path_original",
            "/mir_path_original",
            "/laser_profile_offset_cmd_vel",
        ]
        self.rosbag_enabled = {t: True for t in self.rosbag_topics}
        self.rosbag_process = None


        if not rospy.core.is_initialized():
            rospy.init_node("additive_manufacturing_gui", anonymous=True, disable_signals=True)

        # Subscriptions and publishers used by the GUI
        rospy.Subscriber('/path_index', Int32, self._path_idx_cb, queue_size=10)
        self._init_dynamixel_publishers()
        # Receive medians from robot-side node (published by profiles_median_node on the mur)
        self._last_med_base = float('nan')
        self._last_med_map = float('nan')

        def _median_base_cb(msg: Float32):
            try:
                self._last_med_base = float(msg.data)
            except Exception:
                self._last_med_base = float('nan')
            try:
                self.gui.medians.emit(self._last_med_base, self._last_med_map)
            except Exception:
                pass

        def _median_map_cb(msg: Float32):
            try:
                self._last_med_map = float(msg.data)
            except Exception:
                self._last_med_map = float('nan')
            try:
                self.gui.medians.emit(self._last_med_base, self._last_med_map)
            except Exception:
                pass

        try:
            rospy.Subscriber('/profiles/median_base', Float32, _median_base_cb, queue_size=1)
            rospy.Subscriber('/profiles/median_map', Float32, _median_map_cb, queue_size=1)
        except Exception as e:
            rospy.logwarn(f"Failed to subscribe to median topics: {e}")

        # Subscribe to /rosout to forward log messages into the GUI
        try:
            rospy.Subscriber('/rosout', Log, self._rosout_cb, queue_size=50)
        except Exception as e:
            rospy.logwarn(f"Failed to subscribe to /rosout: {e}")
        
    def _launch_process(self, command, shell=False, **kwargs):
        return _popen_with_debug(command, self.gui, shell=shell, **kwargs)

    def _launch_in_terminal(self, title: str, command: str):
        cmd = ["terminator", f"--title={title}", "-x", f"{command}; exec bash"]
        return self._launch_process(cmd)

    def _debug_prefix_for_workspace(self, workspace: Optional[str] = None) -> str:
        workspace = workspace or self.gui.get_workspace_name()
        return _remote_debug_prefix(self.gui, workspace)

    def init_override_velocity_slider(self):
        self.velocity_override_pub = rospy.Publisher('/velocity_override', Float32, queue_size=10, latch=True)
        self.gui.override_slider.valueChanged.connect(
            lambda value: self.gui.override_value_label.setText(f"{value}%")
            # publish to /velocity_override as well:
            or self.velocity_override_pub.publish(value / 100.0)  # Convert to a float between 0.0 and 1.0
        )

    def init_nozzle_override_slider(self):
        """Wire the nozzle height override slider to /nozzle_height_override."""
        self.nozzle_height_override_pub = rospy.Publisher('/nozzle_height_override', Float32, queue_size=10, latch=True)
        slider = getattr(self.gui, 'nozzle_override_slider', None)
        label = getattr(self.gui, 'nozzle_override_value_label', None)
        if slider is None or label is None:
            return

        def _handle_change(raw_value: int):
            meters = raw_value / 1000.0  # slider uses millimeters
            label.setText(f"{raw_value:.1f} mm")
            self.nozzle_height_override_pub.publish(Float32(meters))

        slider.valueChanged.connect(_handle_change)
        _handle_change(slider.value())
    
    def _path_idx_cb(self, msg: Int32):
        self.current_index = msg.data
        self.gui.path_idx.emit(self.current_index)  # Update the GUI with the new index

    def start_roscore(self):
        """Starts roscore on the roscore PC."""
        workspace = self.gui.get_workspace_name()
        debug_prefix = self._debug_prefix_for_workspace(workspace)
        command = (
            "ssh -t -t roscore 'source ~/.bashrc; source /opt/ros/noetic/setup.bash; "
            f"{debug_prefix}roscore; exec bash'"
        )
        self._launch_in_terminal("Roscore Terminal", command)

    def start_mocap(self):
        """Starts the motion capture system on the roscore PC."""
        workspace = self.gui.get_workspace_name()
        debug_prefix = self._debug_prefix_for_workspace(workspace)
        command = (
            "ssh -t -t roscore 'source ~/.bashrc; source /opt/ros/noetic/setup.bash; "
            "source ~/catkin_ws/devel/setup.bash; "
            f"{debug_prefix}roslaunch launch_mocap mocap_launch.launch; exec bash'"
        )
        self._launch_in_terminal("Mocap", command)

    def start_sync(self):
        """Starts file synchronization between workspace and selected robots."""
        selected_robots = self.gui.get_selected_robots()
        self.workspace_name = self.gui.get_workspace_name()
        self.gui.btn_sync.setStyleSheet("background-color: lightgreen;")  # Mark sync as active
        
        for robot in selected_robots:
            command = f"while inotifywait -r -e modify,create,delete,move ~/{self.workspace_name}/src; do \n" \
                      f"rsync --delete -avzhe ssh ~/{self.workspace_name}/src rosmatch@{robot}:~/{self.workspace_name}/ \n" \
                      "done"
            self._launch_in_terminal(f"Sync to {robot}", command)

    def update_button_status(self):
        # --- Parser + Keyence status ---
        node_cache = self._get_rosnode_list()
        mir = self.is_ros_node_running("/retrieve_and_publish_mir_path", node_cache)
        ur = self.is_ros_node_running("/retrieve_and_publish_ur_path", node_cache)
        key = self.is_ros_node_running("/keyence_ljx_profile_node", node_cache)
        tgt = self.is_ros_node_running("/target_broadcaster", node_cache)
        laser = self.is_ros_node_running("/profile_orthogonal_controller", node_cache)
        mocap = self.is_ros_node_running("/qualisys", node_cache)

        # --- Button coloring ---
        self.gui.btn_parse_mir.setStyleSheet("background-color: lightgreen;" if mir else "background-color: lightgray;") if hasattr(self.gui,"btn_parse_mir") else None
        self.gui.btn_parse_ur.setStyleSheet("background-color: lightgreen;" if ur else "background-color: lightgray;") if hasattr(self.gui,"btn_parse_ur") else None
        self.gui.btn_keyence.setStyleSheet("background-color: lightgreen;" if key else "background-color: lightgray;") if hasattr(self.gui,"btn_keyence") else None
        self.gui.btn_target_broadcaster.setStyleSheet("background-color: lightgreen;" if tgt else "background-color: lightgray;") if hasattr(self.gui,"btn_target_broadcaster") else None
        self.gui.btn_laser_ctrl.setStyleSheet("background-color: lightgreen;" if laser else "background-color: lightgray;") if hasattr(self.gui,"btn_laser_ctrl") else None
        self.gui.btn_mocap.setStyleSheet("background-color: lightgreen;" if mocap else "background-color: lightgray;") if hasattr(self.gui,"btn_mocap") else None

        # --- Driver status (UR hardware interface nodes) ---
        driver_nodes = self._driver_node_names(
            robots=self.gui.get_selected_robots(),
            urs=self.gui.get_selected_urs(),
        )
        driver_total = len(driver_nodes)
        driver_active = sum(1 for node in driver_nodes if node in node_cache)
        if hasattr(self.gui, "btn_launch_drivers"):
            if driver_total == 0 or driver_active == 0:
                driver_color = "background-color: lightgray;"
            elif driver_active == driver_total:
                driver_color = "background-color: lightgreen;"
            else:
                driver_color = "background-color: orange;"
            self.gui.btn_launch_drivers.setStyleSheet(driver_color)

        # --- Auto-start target broadcaster ---
        if mir and ur and not tgt: 
            print("Auto-starting /target_broadcaster…"); _popen_with_debug(f"roslaunch print_hw target_broadcaster.launch initial_path_index:={self.gui.idx_spin.value()}", self.gui, shell=True)

    def _get_rosnode_list(self):
        """Return the current ROS node names as a set for quick membership checks."""
        try:
            output = subprocess.check_output(["rosnode", "list"], text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return set()
        return {line.strip() for line in output.splitlines() if line.strip()}

    def is_ros_node_running(self, node_name, node_cache=None):
        """Checks if a specific ROS node is running by using `rosnode list`."""
        nodes = node_cache if node_cache is not None else self._get_rosnode_list()
        return node_name in nodes

    def _driver_node_names(self, robots=None, urs=None):
        """Return the UR driver node names to monitor for the given robot/UR selection."""
        robot_names = robots if robots else list(getattr(self.gui, "robots", {}).keys())
        if not robot_names:
            return []
        ur_prefixes = urs if urs else ["UR10_l", "UR10_r"]
        names = []
        for robot in robot_names:
            for ur in ur_prefixes:
                names.append(f"/{robot}/{ur}/ur_hardware_interface")
        return names
        
    def make_ur_callback(self, robot):
        def callback(msg):
            if robot not in self.battery_states:
                self.battery_states[robot] = {}
            self.battery_states[robot]["ur"] = msg.data
            self.update_battery_display(robot)
        return callback

    def make_mir_callback(self, robot):
        def callback(msg):
            if robot not in self.battery_states:
                self.battery_states[robot] = {}
            self.battery_states[robot]["mir"] = msg.percentage * 100  # falls 0.0–1.0
            self.update_battery_display(robot)
        return callback

    def update_battery_display(self, robot):
        mir_label, ur_label = self.gui.battery_labels.get(robot, (None, None))
        if not mir_label or not ur_label:
            return

        state = self.battery_states.get(robot, {})

        if "mir" in state:
            val = state["mir"]
            mir_label.setText(f"MiR: {val:.0f}%")
            mir_label.setStyleSheet(f"background-color: {self.get_color(val)}; padding: 2px;")

        if "ur" in state:
            val = state["ur"]
            ur_label.setText(f"UR: {val:.0f}%")
            ur_label.setStyleSheet(f"background-color: {self.get_color(val)}; padding: 2px;")


    def check_and_subscribe_battery(self):
        if not rospy.core.is_initialized():
            rospy.init_node("battery_status_gui", anonymous=True, disable_signals=True)

        for robot in self.gui.get_selected_robots():
            if robot in self.active_battery_subs:
                continue  # bereits abonniert

            rospy.Subscriber(f"/{robot}/bms_status/SOC", Float32, self.make_ur_callback(robot), queue_size=1)
            rospy.Subscriber(f"/{robot}/battery_state", BatteryState, self.make_mir_callback(robot))
            self.active_battery_subs.add(robot)
            print(f"🔌 Subscribed to battery topics for {robot}")

    def get_color(self, val):
        if val >= 50:
            return "lightgreen"
        elif val >= 20:
            return "orange"
        else:
            return "red"

    def launch_keyence_scanner(self):
        """Launches the Keyence scanner on the robots PC."""
        selected_robots = self.gui.get_selected_robots()
        if not selected_robots:
            print("No robot selected. Skipping Keyence scanner launch.")
            return

        _run_remote_commands(
            self.gui,
            "Launching Keyence scanner",
            ["roslaunch laser_scanner_tools keyence_scanner_ljx8000.launch"],
            use_workspace_debug=True,
            target_robots=selected_robots,
        )

    def launch_laser_orthogonal_controller(self):
        """Launches the Keyence scanner on the robots PC."""
        selected_robots = self.gui.get_selected_robots()
        if not selected_robots:
            print("No robot selected. Skipping orthogonal controller launch.")
            return

        _run_remote_commands(
            self.gui,
            "Launching orthogonal controller",
            ["roslaunch laser_scanner_tools profile_orthogonal_controller.launch"],
            use_workspace_debug=True,
            target_robots=selected_robots,
        )

    def launch_strand_center_app(self):
        """Launch the DepthAI strand-center app on the selected robots over SSH."""
        selected_robots = self.gui.get_selected_robots()
        if not selected_robots:
            print("No robot selected. Skipping strand-center launch.")
            return

        workspace = self.gui.get_workspace_name()
        script_rel = "src/match_additive_manufacturing/camera_oak/strand-center-app/main.py"

        for robot in selected_robots:
            remote_script = f"~/{workspace}/{script_rel}"
            debug_prefix = self._debug_prefix_for_workspace(workspace)
            command = (
                f"ssh -t -t {robot} '"
                "source ~/.bashrc; "
                "export ROS_MASTER_URI=http://roscore:11311/; "
                "source /opt/ros/noetic/setup.bash; "
                f"source ~/{workspace}/devel/setup.bash; "
                f"{debug_prefix}python3 {remote_script}; "
                "exec bash'"
            )

            self._launch_in_terminal(f"StrandCenter {robot}", command)

    def toggle_rosbag_record(self):
        # --- Stop recording ---
        if self.rosbag_process and self.rosbag_process.poll() is None:
            print("Stopping rosbag (SIGINT)…")
            try:
                self.rosbag_process.send_signal(subprocess.signal.SIGINT)
                self.rosbag_process.wait(timeout=4)
            except Exception:
                print("Rosbag didn't stop cleanly. Sending SIGTERM…")
                self.rosbag_process.terminate()
                try:
                    self.rosbag_process.wait(timeout=2)
                except Exception:
                    print("Force killing rosbag…")
                    self.rosbag_process.kill()

            self.rosbag_process = None
            self.gui.btn_rosbag_record.setStyleSheet("background-color: lightgray;")
            return

        # --- Start recording ---
        topics = [t for t, ok in self.rosbag_enabled.items() if ok]
        cmd = ["rosbag", "record", "-O", "recording", "--lz4"] + topics
        self.rosbag_process = subprocess.Popen(cmd)
        self.gui.btn_rosbag_record.setStyleSheet("background-color: red; color: white;")
        
    # -------------------- Dynamixel Driver & Servo Targets --------------------
    def start_dynamixel_driver(self):
        """Start the Dynamixel servo driver on the selected robots over SSH (new terminal per robot)."""
        selected_robots = self.gui.get_selected_robots()
        if not selected_robots:
            print("No robot selected. Skipping Dynamixel start.")
            return
        workspace = self.gui.get_workspace_name()
        for robot in selected_robots:
            debug_prefix = self._debug_prefix_for_workspace(workspace)
            command = (
                f"ssh -t -t {robot} '"
                f"source ~/.bashrc; "
                f"export ROS_MASTER_URI=http://roscore:11311/; "
                f"source /opt/ros/noetic/setup.bash; "
                f"source ~/{workspace}/devel/setup.bash; "
                f"{debug_prefix}roslaunch dynamixel_match dynamixel_motors.launch; exec bash'"
            )
            self._launch_in_terminal(f"Dynamixel {robot}", command)

    def stop_dynamixel_driver(self):
        """Stop the Dynamixel servo driver on the selected robots over SSH."""
        selected_robots = self.gui.get_selected_robots()
        if not selected_robots:
            print("No robot selected. Skipping Dynamixel stop.")
            return
        workspace = self.gui.get_workspace_name()
        for robot in selected_robots:
            # Kill by rosnode and fall back to process patterns
            remote_cmd = (
                "source ~/.bashrc; "
                "export ROS_MASTER_URI=http://roscore:11311/; "
                "source /opt/ros/noetic/setup.bash; "
                f"source ~/{workspace}/devel/setup.bash; "
                "rosnode kill /servo_driver || true; "
                "pkill -f dynamixel_motors.launch || true; "
                "pkill -f servo_driver.py || true; "
                "pkill -f dynamixel_workbench_controllers || true;"
            )
            self._launch_process(f"ssh -t -t {robot} '{remote_cmd}'", shell=True)

    def _init_dynamixel_publishers(self):
        """Initialize publishers for left/right servo target topics (Int16)."""
        try:
            self.servo_left_pub = rospy.Publisher('/servo_target_pos_left', Int16, queue_size=10)
            self.servo_right_pub = rospy.Publisher('/servo_target_pos_right', Int16, queue_size=10)
        except Exception as e:
            rospy.logwarn(f"Failed to create Dynamixel target publishers: {e}")

    def publish_servo_targets(self, left_value: int, right_value: int):
        """Publish target positions for both servos as Int16 (0..4095).
        Values are forwarded as std_msgs/Int16 on /servo_target_pos_left and /servo_target_pos_right.
        """
        if not hasattr(self, 'servo_left_pub'):
            self._init_dynamixel_publishers()
        try:
            # Coerce to servo range 0..4095
            left_int = max(min(int(round(left_value)), 4095), 0)
            right_int = max(min(int(round(right_value)), 4095), 0)
            self.servo_left_pub.publish(Int16(left_int))
            self.servo_right_pub.publish(Int16(right_int))
            rospy.loginfo(f"Published servo targets L={left_int}, R={right_int}")
        except Exception as e:
            rospy.logerr(f"Failed to publish servo targets: {e}")

    def _rosout_cb(self, msg: Log):
        """
        Callback for /rosout (rosgraph_msgs/Log).
        Forwards a compact text line to the Qt GUI via signal.
        """
        try:
            text = getattr(msg, "msg", "")

            # --- Blacklist bestimmter Meldungen ---
            blacklist = [
                "The complete state of the robot is not yet known",
                "Battery:",
                "Unable to transform object from frame",
                # hier ggf. weitere Phrasen ergänzen
            ]
            for phrase in blacklist:
                if phrase in text:
                    return  # einfach ignorieren

            # Map level to readable string (if constants exist)
            level_map = {
                getattr(Log, "DEBUG", 1): "DEBUG",
                getattr(Log, "INFO", 2): "INFO",
                getattr(Log, "WARN", 4): "WARN",
                getattr(Log, "ERROR", 8): "ERROR",
                getattr(Log, "FATAL", 16): "FATAL",
            }
            level = level_map.get(msg.level, str(msg.level))
            node_name = getattr(msg, "name", "?")

            if hasattr(self.gui, "ros_log_signal"):
                self.gui.ros_log_signal.emit(level, node_name, text)
        except Exception as e:
            rospy.logwarn(f"Error while forwarding /rosout message: {e}")



def launch_ros(gui, package, launch_file):
    selected_robots = gui.get_selected_robots()
    robot_names_str = "[" + ",".join(f"'{r}'" for r in selected_robots) + "]"

    command = f"roslaunch {package} {launch_file} robot_names:={robot_names_str}"
    print(f"Executing: {command}")
    _popen_with_debug(command, gui, shell=True)

def start_status_update(gui):
    threading.Thread(target=update_status, args=(gui,), daemon=True).start()

def update_status(gui):
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()
    active_counts = {"force_torque_sensor_controller": 0, "twist_controller": 0, "arm_controller": 0, "admittance": 0}
    total_count = len(selected_robots) * len(selected_urs)
    
    for robot in selected_robots:
        for ur in selected_urs:
            service_name = f"/{robot}/{ur}/controller_manager/list_controllers"
            try:
                output = subprocess.check_output(f"rosservice call {service_name}", shell=True).decode()
                controllers = yaml.safe_load(output).get("controller", [])
                for controller in controllers:
                    if controller.get("state") == "running":
                        active_counts[controller["name"]] = active_counts.get(controller["name"], 0) + 1
            except Exception as e:
                rospy.logwarn(f"Error checking controllers for {robot}/{ur}: {e}")
    
    status_text = """
    Force/Torque Sensor: {}/{} {}
    Twist Controller: {}/{} {}
    Arm Controller: {}/{} {}
    Admittance Controller: {}/{} {}
    """.format(
        active_counts["force_torque_sensor_controller"], total_count, get_status_symbol(active_counts["force_torque_sensor_controller"], total_count),
        active_counts["twist_controller"], total_count, get_status_symbol(active_counts["twist_controller"], total_count),
        active_counts["arm_controller"], total_count, get_status_symbol(active_counts["arm_controller"], total_count),
        active_counts["admittance"], total_count, get_status_symbol(active_counts["admittance"], total_count),
    )
    
    gui.status_label.setText(status_text)

def get_status_symbol(active, total):
    if active == total:
        return "✅"
    elif active > 0:
        return "⚠️"
    return "❌"

def open_rviz(gui):
    command = "roslaunch print_gui launch_rviz.launch"
    _popen_with_debug(command, gui, shell=True)

def turn_on_arm_controllers(gui):
    """Turns on all arm controllers for the selected robots."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    ur_prefixes_str = _format_ros_list_arg(selected_urs)

    for robot in selected_robots:
        robot_list_str = _format_ros_list_arg([robot])
        _run_remote_commands(
            gui,
            f"Turning on arm controllers",
            [
                (
                    f"roslaunch print_gui turn_on_all_arm_controllers.launch "
                    f"robot_names:={robot_list_str} UR_prefixes:={ur_prefixes_str}"
                )
            ],
            use_workspace_debug=True,
            target_robots=[robot],
        )

def turn_on_twist_controllers(gui):
    """Turns on all twist controllers for the selected robots."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    ur_prefixes_str = _format_ros_list_arg(selected_urs)

    for robot in selected_robots:
        robot_list_str = _format_ros_list_arg([robot])
        _run_remote_commands(
            gui,
            "Turning on twist controllers",
            [
                (
                    f"roslaunch print_gui turn_on_all_twist_controllers.launch "
                    f"robot_names:={robot_list_str} UR_prefixes:={ur_prefixes_str}"
                )
            ],
            use_workspace_debug=True,
            target_robots=[robot],
        )

def enable_all_urs(gui):
    """Enables all UR robots for the selected configurations."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping launch.")
        return

    ur_prefixes_str = _format_ros_list_arg(selected_urs)

    for robot in selected_robots:
        robot_list_str = _format_ros_list_arg([robot])
        _run_remote_commands(
            gui,
            "Enabling URs",
            [
                (
                    f"roslaunch print_gui enable_all_URs.launch "
                    f"robot_names:={robot_list_str} UR_prefixes:={ur_prefixes_str}"
                )
            ],
            use_workspace_debug=True,
            target_robots=[robot],
        )


def launch_drivers(gui):
    """SSH into the selected robots and start the drivers in separate terminals."""
    selected_robots = gui.get_selected_robots()
    workspace_name = gui.get_workspace_name()

    if not selected_robots:
        print("No robots selected. Skipping driver launch.")
        return

    if not hasattr(gui, "_driver_processes"):
        gui._driver_processes = []
    gui._driver_processes = [p for p in gui._driver_processes if p and p.poll() is None]
    gui._driver_robots = list(set(selected_robots))

    for robot in selected_robots:
        workspace = workspace_name
        selected_urs = gui.get_selected_urs()
        launch_suffix = ""
        if "UR10_l" in selected_urs:
            launch_suffix += " launch_ur_l:=true"
        else:
            launch_suffix += " launch_ur_l:=false"
        if "UR10_r" in selected_urs:
            launch_suffix += " launch_ur_r:=true"
        else:
            launch_suffix += " launch_ur_r:=false"

        try:
            offset_values = gui.get_tcp_offset_sixd()
        except AttributeError:
            offset_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tcp_offset_literal = "[" + ",".join(f"{value:.6f}" for value in offset_values) + "]"

        debug_prefix = _remote_debug_prefix(gui, workspace)
        command = (
            f"ssh -t -t {robot} '"
            "source ~/.bashrc; "
            "export ROS_MASTER_URI=http://roscore:11311/; "
            "source /opt/ros/noetic/setup.bash; "
            f"source ~/{workspace}/devel/setup.bash; "
            f"{debug_prefix}roslaunch mur_launch_hardware {robot}.launch{launch_suffix} "
            f"tcp_offset:=\\\"{tcp_offset_literal}\\\"; "
            "exec bash'"
        )

        proc = _popen_with_debug([
            "terminator",
            f"--title=Driver {robot}",
            "-x",
            f"{command}; exec bash"
        ], gui)
        if proc is not None:
            gui._driver_processes.append(proc)

def quit_drivers(gui=None):
    """Terminates all running driver sessions and closes terminals."""
    print("Stopping all driver sessions...")
    processes = []
    if gui is not None and hasattr(gui, "_driver_processes"):
        processes = [p for p in gui._driver_processes if p and p.poll() is None]

    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as exc:
            rospy.logwarn(f"Failed to terminate driver terminal: {exc}")

    if gui is not None:
        gui._driver_processes = [p for p in getattr(gui, "_driver_processes", []) if p and p.poll() is None]

    # Ask the robots themselves to stop the driver launch files.
    robots = []
    if gui is not None:
        robots = gui.get_selected_robots()
        if not robots:
            robots = getattr(gui, "_driver_robots", [])

    if gui is not None and robots:
        def _remote_kill(robot):
            kill_patterns = [
                f"pkill -f 'mur_launch_hardware {robot}\\.launch' || true",
                "pkill -f ur_robot_driver || true",
                "pkill -f roslaunch || true",
            ]
            return "; ".join(kill_patterns)

        _run_remote_commands(
            gui,
            "Stopping remote driver launch",
            [_remote_kill],
            use_workspace_debug=True,
            target_robots=robots,
        )

    # Best-effort fallback in case processes were started outside this session.
    try:
        _popen_with_debug("pkill -f 'terminator.*Driver'", gui, shell=True)
    except Exception as e:
        print(f"Error stopping processes: {e}")

def move_to_home_pose(gui, UR_prefix):
    """Moves the selected robots to the initial pose with the correct namespace and move_group_name."""
    selected_robots = gui.get_selected_robots()

    # Set move_group_name based on UR_prefix
    move_group_name = "UR_arm_l" if UR_prefix == "UR10_l" else "UR_arm_r"

    for robot in selected_robots:
        # Special case for mur620c with UR10_r
        if robot == "mur620c" and UR_prefix == "UR10_r":
            home_position = "Home_custom"
        elif robot in ["mur620a", "mur620b"]:
            home_position = "Home_custom"
        else:  # Default case for mur620c, mur620d
            home_position = "Home_custom"

        # ROS launch command with namespace
        command = f"ROS_NAMESPACE={robot} roslaunch ur_utilities move_UR_to_home_pose.launch tf_prefix:={robot} UR_prefix:={UR_prefix} home_position:={home_position} move_group_name:={move_group_name}"
        print(f"Executing: {command}")
        _popen_with_debug(command, gui, shell=True)

def parse_mir_path(gui):
    selected_robots = gui.get_selected_robots()
    if not selected_robots:
        print("No robot selected. Skipping MiR path parsing.")
        return

    _run_remote_commands(
        gui,
        "Parsing MiR path",
        ["roslaunch parse_mir_path parse_mir_path.launch"],
        use_workspace_debug=True,
        target_robots=selected_robots,
    )

def parse_ur_path(gui):
    selected_robots = gui.get_selected_robots()
    if not selected_robots:
        print("No robot selected. Skipping UR path parsing.")
        return

    _run_remote_commands(
        gui,
        "Parsing UR path",
        ["roslaunch parse_ur_path parse_ur_path.launch"],
        use_workspace_debug=True,
        target_robots=selected_robots,
    )

def move_mir_to_start_pose(gui):
    """Moves the MIR robot to the start pose."""
    selected_robots = gui.get_selected_robots()
    if selected_robots is None:
        print("MIR robot not selected. Skipping move to start pose.")
        return

    # Ensure only one MIR robot is selected
    if len(selected_robots) != 1:
        print("Please select only the MIR robot to move to the start pose.")
        return
    command = f"roslaunch move_mir_to_start_pose move_mir_to_start_pose.launch robot_name:={selected_robots[0]} initial_path_index:={gui.idx_spin.value()}"
    print(f"Executing: {command}")
    _popen_with_debug(command, gui, shell=True)

def move_ur_to_start_pose(gui):
    """Moves the UR robot to the start pose."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()

    if not selected_robots or not selected_urs:
        print("No robots or URs selected. Skipping move to start pose.")
        return

    # Ensure only one UR and one mir robot is selected
    if len(selected_robots) != 1 or len(selected_urs) != 1:
        print("Please select exactly one UR and one MIR robot to move to the start pose.")
        return

    # ToDo: to risky to use for now
    if selected_urs[0] == "UR10_l":
        move_group_name = "UR_arm_l"
        planning_group = "UR10_l/tool0"
    else:
        move_group_name = "UR_arm_r"
        planning_group = "UR10_r/tool0"

    spray_distance = gui.get_spray_distance()

    for robot in selected_robots:
        for ur in selected_urs:
            command = f"roslaunch move_ur_to_start_pose move_ur_to_start_pose.launch robot_name:={robot} initial_path_index:={gui.idx_spin.value()} spray_distance:={spray_distance}"
            print(f"Executing: {command}")
            _popen_with_debug(command, gui, shell=True)

def mir_follow_trajectory(gui):
    """Moves the MIR robot along a predefined trajectory."""
    selected_robots = gui.get_selected_robots()
    if not selected_robots:
        print("No MIR robot selected. Skipping follow trajectory.")
        return
    # Ensure only one MIR robot is selected
    if len(selected_robots) != 1:
        print("Please select only the MIR robot to follow the trajectory.")
        return
    _run_remote_commands(
        gui,
        "Launching MiR trajectory follower",
        [lambda robot: f"roslaunch mir_trajectory_follower mir_trajectory_follower.launch robot_name:={robot}"],
        use_workspace_debug=True,
        target_robots=selected_robots,
    )

def increment_path_index(gui):
    """Increments the path index for the MIR robot."""
    selected_robots = gui.get_selected_robots()
    if not selected_robots:
        print("No robot selected. Skipping path index increment.")
        return

    initial_idx = gui.idx_spin.value()
    _run_remote_commands(
        gui,
        "Incrementing path index",
        [f"roslaunch print_gui increment_path_index.launch initial_path_index:={initial_idx}"],
        use_workspace_debug=True,
        target_robots=selected_robots,
    )

    # if not rospy.core.is_initialized():
    #     rospy.init_node("additive_manufacturing_gui", anonymous=True, disable_signals=True)
    # rospy.Subscriber('/path_index', Int32, gui.ros_interface._path_idx_cb, queue_size=10)


def stop_mir_motion(gui):
    """Stops MiR motion on the selected robots via SSH."""
    _run_remote_commands(
        gui,
        "Stopping MiR motion",
        [
            "pkill -f mir_trajectory_follower || true",
            "pkill -f increment_path_index || true",
        ],
    )


def stop_ur_motion(gui):
    """Stops UR motion on the selected robots via SSH."""
    _run_remote_commands(
        gui,
        "Stopping UR motion",
        [
            "pkill -f 'ur_direction_controller|orthogonal_error_correction|move_ur_to_start_pose|ur_vel_induced_by_mir|world_twist_in_mir|twist_combiner|ur_yaw_controller' || true",
        ],
    )


def stop_all_but_drivers(gui):
    """Stops common non-driver processes on the selected robots via SSH."""
    _run_remote_commands(
        gui,
        "Stopping non-driver processes",
        [
            "pkill -f mir_trajectory_follower || true",
            "pkill -f increment_path_index || true",
            "pkill -f path_index_advancer || true",
            "pkill -f complete_ur_trajectory_follower_ff_only || true",
            "pkill -f 'ur_direction_controller|orthogonal_error_correction|move_ur_to_start_pose|ur_vel_induced_by_mir|world_twist_in_mir|twist_combiner|ur_yaw_controller' || true",
            "pkill -f keyence_scanner_ljx8000 || true",
            "pkill -f profile_orthogonal_controller || true",
            "pkill -f strand-center-app || true",
            "pkill -f parse_mir_path || true",
            "pkill -f parse_ur_path || true",
            "pkill -f target_broadcaster || true",
            "pkill -f move_mir_to_start_pose || true",
            "pkill -f move_ur_to_start_pose || true",
        ],
    )

def ur_follow_trajectory(gui, ur_follow_settings: dict):
    """Moves the UR robot along a predefined trajectory."""
    selected_robots = gui.get_selected_robots()
    selected_urs = gui.get_selected_urs()
    metric = ur_follow_settings.get("idx_metric")
    threshold = ur_follow_settings.get("threshold")
    initial_path_index = gui.idx_spin.value()
    spray_distance = gui.get_spray_distance()
    rospy.loginfo(f"Selected metric: {metric}")

    if not selected_robots or not selected_urs:
        print("No UR robot selected. Skipping follow trajectory.")
        return

    # Ensure only one UR and one MIR robot is selected
    if len(selected_robots) != 1 or len(selected_urs) != 1:
        print("Please select exactly one UR and one MIR robot to follow the trajectory.")
        return

    ur = selected_urs[0]

    cmd_template = (
        "roslaunch print_hw complete_ur_trajectory_follower_ff_only.launch "
        "robot_name:={robot} prefix_ur:={ur}/ metric:='{metric}' "
        "threshold:={threshold} initial_path_index:={initial_path_index} "
        "nozzle_height_default:={spray_distance}"
    )

    _run_remote_commands(
        gui,
        "Launching UR trajectory follower",
        [
            lambda robot, template=cmd_template: template.format(
                robot=robot,
                ur=ur,
                metric=metric,
                threshold=threshold,
                initial_path_index=initial_path_index,
                spray_distance=spray_distance,
            )
        ],
        use_workspace_debug=True,
        target_robots=selected_robots,
    )

def stop_idx_advancer(gui):
    """Stops the path index advancer nodes on the selected robots via SSH."""
    _run_remote_commands(
        gui,
        "Stopping path index advancer",
        [
            "pkill -f path_index_advancer || true",
            "pkill -f increment_path_index || true",
        ],
    )


def target_broadcaster(gui):
    """Broadcasts Target Poses."""
    command = f"roslaunch print_hw target_broadcaster.launch initial_path_index:={gui.idx_spin.value()}"
    print(f"Executing: {command}")
    _popen_with_debug(command, gui, shell=True)